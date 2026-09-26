#pragma once
namespace ait::native_fusion {
// Direct TMEM-to-register epilogue for source-free outputs.
// No shared-memory output allocation or TMA output pipeline is needed.
template <class Base>
struct DirectEpilogue {
  static_assert(cute::is_void_v<typename Base::ElementC>);
  using ThreadEpilogueOp = typename Base::ThreadEpilogueOp;
  using FusionCallbacks = typename Base::FusionCallbacks;
  using EpilogueTile = typename Base::EpilogueTile;
  using ElementC = typename Base::ElementC;
  using ElementD = typename Base::ElementD;
  using StrideC = typename Base::StrideC;
  using StrideD = typename Base::StrideD;
  using Arguments = typename Base::Arguments;
  using Params = typename Base::Params;
  using LoadPipeline = typename Base::LoadPipeline;
  using LoadPipelineState = typename Base::LoadPipelineState;
  using StorePipeline = typename Base::StorePipeline;
  using StorePipelineState = typename Base::StorePipelineState;
  using PipelineStorage = typename Base::PipelineStorage;
  struct TensorStorage {};
  using SharedStorage = TensorStorage;
  static constexpr int ThreadCount = 128, NumAccumulatorMtxs = 1,
                       TmaTransactionBytes = 0;
  typename FusionCallbacks::Params direct_params;
  CUTLASS_DEVICE DirectEpilogue(Params const& p, TensorStorage&)
      : direct_params(p.thread) {}
  template <class... A>
  static auto to_underlying_arguments(A&&... a) {
    return Base::to_underlying_arguments(static_cast<A&&>(a)...);
  }
  template <class... A>
  static bool can_implement(A&&... a) {
    return Base::can_implement(static_cast<A&&>(a)...);
  }
  template <class... A>
  static size_t get_workspace_size(A&&... a) {
    return Base::get_workspace_size(static_cast<A&&>(a)...);
  }
  template <class... A>
  static auto initialize_workspace(A&&... a) {
    return Base::initialize_workspace(static_cast<A&&>(a)...);
  }
  template <class T>
  static constexpr int get_load_pipe_increment(T) {
    return 0;
  }
  CUTLASS_DEVICE bool is_producer_load_needed() const {
    return false;
  }
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const&) {}
  template <bool Reuse = false, class... A>
  CUTLASS_DEVICE auto load(LoadPipeline, LoadPipelineState state, A&&...) {
    return state;
  }
  template <class... A>
  CUTLASS_DEVICE void load_tail(A&&...) {}
  template <
      bool ReuseTmem = false,
      class AP,
      class AS,
      class PS,
      class CT,
      class CC,
      class MT,
      class MMA,
      class AE,
      class AL>
  CUTLASS_DEVICE auto store(
      LoadPipeline,
      LoadPipelineState ls,
      StorePipeline,
      StorePipelineState ss,
      AP ap,
      AS as,
      PS problem,
      CT cta,
      CC coord,
      MT,
      MMA,
      cute::Tensor<AE, AL> accum,
      TensorStorage&) {
    using namespace cute;
    auto [M, N, K, L] = problem;
    auto [m, n, k, l] = coord;
    auto tacc = accum(make_coord(_, _), _0{}, _0{});
    auto tiles = flat_divide(tacc, EpilogueTile{});
    auto tiled =
        make_tmem_copy(typename Base::CopyOpT2R{}, tiles(_, _, _0{}, _0{}));
    int tid = threadIdx.x % 128;
    auto thread = tiled.get_slice(tid);
    auto src = thread.partition_S(tiles);
    auto id = local_tile(
        make_identity_tensor(make_shape(M, N)),
        take<0, 2>(cta),
        make_coord(m, n));
    auto coordinates = thread.partition_D(flat_divide(id, EpilogueTile{}));
    auto registers =
        make_tensor<float>(shape(coordinates(_, _, _, _0{}, _0{})));
    auto fragments = recast<cutlass::Array<float, 64>>(coalesce(registers));
    static_assert(size(fragments) == 1);
    typename Base::FusionCallbacks::template Callbacks<decltype(coordinates)>
        callback(coordinates, direct_params, M);
    ap.consumer_wait(as);
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<3>(tiles); ++ni) {
      int en = ni;
      copy(tiled, src(_, _, _, _0{}, en), registers);
      if ((ReuseTmem && ni == 0) || (!ReuseTmem && ni == size<3>(tiles) - 1)) {
        cutlass::arch::fence_view_async_tmem_load();
        ap.consumer_release(as);
        ++as;
      }
      callback.visit(fragments(0), 0, 0, en);
    }
    return make_tuple(ls, ss, as);
  }
  template <class CT>
  CUTLASS_DEVICE auto store_tail(
      LoadPipeline,
      LoadPipelineState ls,
      StorePipeline,
      StorePipelineState ss,
      CT) {
    return cute::make_tuple(ls, ss);
  }
};

} // namespace ait::native_fusion
