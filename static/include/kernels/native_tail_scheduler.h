#pragma once
#include <algorithm>
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
namespace ait::native_fusion {
struct TailSchedulerTag {};
// Single-CTA clusters, static persistence, and serpentine groups without padded
// tiles.
class TailScheduler
    : public cutlass::gemm::kernel::detail::StaticPersistentTileScheduler100 {
  using Base = cutlass::gemm::kernel::detail::StaticPersistentTileScheduler100;
  uint64_t current_ = 0, stride_ = 0;

 public:
  struct Params : Base::Params {
    cutlass::FastDivmodU64 group_tiles, group_size, tail_size;
    uint64_t regular_groups = 0, slow = 0;
  };
  Params exact_;
  CUTLASS_HOST_DEVICE TailScheduler() = default;
  CUTLASS_DEVICE TailScheduler(
      CLCResponse* response,
      Params const& p,
      dim3 block)
      : Base(response, p, block), exact_(p) {
    if (p.raster_order_ == RasterOrder::AlongN)
      current_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * gridDim.x;
    else
      current_ = uint64_t(blockIdx.x) * gridDim.y + blockIdx.y;
    stride_ = uint64_t(gridDim.x) * gridDim.y * gridDim.z;
  }
  CUTLASS_DEVICE explicit TailScheduler(Params const& p) : Base(p), exact_(p) {
    if (p.raster_order_ == RasterOrder::AlongN)
      current_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * gridDim.x;
    else
      current_ = uint64_t(blockIdx.x) * gridDim.y + blockIdx.y;
    stride_ = uint64_t(gridDim.x) * gridDim.y * gridDim.z;
  }
  template <class PS, class TS, class CS, class... Rest>
  static Params to_underlying_arguments(
      PS problem,
      TS tile,
      CS cluster,
      cutlass::KernelHardwareInfo const& hw,
      Arguments const& args,
      Rest... rest) {
    static_assert(
        cute::size(CS{}) == 1, "Tail scheduler requires a single-CTA cluster");
    Arguments linear = args;
    linear.max_swizzle_size = 1;
    auto base = Base::to_underlying_arguments(
        problem, tile, cluster, hw, linear, rest...);
    return finish_params(base, args);
  }
  template <class PS, class TS, class AT, class CS, class... Rest>
  static Params to_underlying_arguments(
      PS problem,
      TS tile,
      AT atom,
      CS cluster,
      cutlass::KernelHardwareInfo const& hw,
      Arguments const& args,
      Rest... rest) {
    static_assert(
        cute::size(CS{}) == 1, "Tail scheduler requires a single-CTA cluster");
    Arguments linear = args;
    linear.max_swizzle_size = 1;
    Params p;
    static_cast<Base::Params&>(p) = Base::to_underlying_arguments(
        problem, tile, atom, cluster, hw, linear, rest...);
    return finish_params(p, args);
  }
  static Params finish_params(Base::Params const& base, Arguments const& args) {
    Params p;
    static_cast<Base::Params&>(p) = base;
    uint64_t fast = p.raster_order_ == RasterOrder::AlongM ? p.problem_tiles_m_
                                                           : p.problem_tiles_n_;
    p.slow = p.raster_order_ == RasterOrder::AlongM ? p.problem_tiles_n_
                                                    : p.problem_tiles_m_;
    uint64_t group =
        std::min<uint64_t>(std::max(args.max_swizzle_size, 1), fast);
    p.regular_groups = fast / group;
    p.group_tiles = cutlass::FastDivmodU64(group * p.slow);
    p.group_size = cutlass::FastDivmodU64(group);
    p.tail_size = cutlass::FastDivmodU64(std::max<uint64_t>(fast % group, 1));
    return p;
  }
  CUTLASS_DEVICE WorkTileInfo
  get_current_work_for_linear_idx(uint64_t idx) const {
    if (idx >= exact_.blocks_per_problem_)
      return WorkTileInfo::invalid_work_tile();
    uint64_t batch, within;
    exact_.divmod_batch_(batch, within, idx);
    uint64_t group, offset;
    exact_.group_tiles(group, offset, within);
    uint64_t slow, fast;
    if (group < exact_.regular_groups)
      exact_.group_size(slow, fast, offset);
    else
      exact_.tail_size(slow, fast, offset);
    if (group & 1)
      slow = exact_.slow - 1 - slow;
    fast += group * exact_.group_size.divisor;
    return exact_.raster_order_ == RasterOrder::AlongM
        ? WorkTileInfo{int(fast), int(slow), int(batch), true}
        : WorkTileInfo{int(slow), int(fast), int(batch), true};
  }
  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    return get_current_work_for_linear_idx(current_);
  }
  template <class CS>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(CS) {
    return get_current_work();
  }
  CUTLASS_DEVICE void advance_to_next_work(uint32_t count = 1) {
    current_ += stride_ * count;
  }
  CUTLASS_DEVICE bool is_last_tile(WorkTileInfo&, uint32_t count = 1) const {
    return !get_current_work_for_linear_idx(current_ + stride_ * count)
                .is_valid();
  }
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo) {
    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }
  template <class P, class S>
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo work, P&, S) {
    return fetch_next_work(work);
  }
};
} // namespace ait::native_fusion
namespace cutlass::gemm::kernel::detail {
template <class Tile, class Cluster, uint32_t Stages, class Problem>
struct TileSchedulerSelector<
    ait::native_fusion::TailSchedulerTag,
    cutlass::arch::Sm100,
    Tile,
    Cluster,
    Stages,
    Problem> {
  using Scheduler = ait::native_fusion::TailScheduler;
};
template <class Tile, class Cluster, uint32_t Stages, class Problem>
struct TileSchedulerSelector<
    ait::native_fusion::TailSchedulerTag,
    cutlass::arch::Sm90,
    Tile,
    Cluster,
    Stages,
    Problem> {
  using Scheduler = ait::native_fusion::TailScheduler;
};
} // namespace cutlass::gemm::kernel::detail
