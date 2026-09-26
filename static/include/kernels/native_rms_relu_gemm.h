#pragma once
#include "native_gdc.h"
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "native_epilogue_pipeline.h"
#include "native_sm100_pipeline.h"
#include "native_tail_scheduler.h"
#include "native_row_reduction.h"
#include "native_parallel_producer_sm90.h"
#include "native_cluster_launch.h"
namespace ait::native_fusion::boundary {
using namespace cute;
using namespace cutlass::epilogue::fusion;
using H = cutlass::half_t;
using L = cutlass::layout::RowMajor;
constexpr auto RN = cutlass::FloatRoundStyle::round_to_nearest;

template <class T>
struct Square {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const {
    return cutlass::multiplies<T>{}(x, x);
  }
};
struct NormalizeArguments {
  float eps = 1e-6f;
};
template <class T>
struct Normalize {
  using Arguments = NormalizeArguments;
};
template <class T, int N>
struct Normalize<cutlass::Array<T, N>> {
  using Arguments = NormalizeArguments;
  CUTLASS_DEVICE cutlass::Array<T, N> operator()(
      cutlass::Array<T, N> const& x,
      Arguments const& args) const {
    cutlass::Array<T, N> y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i)
      y[i] = rsqrtf(x[i] / 192.f + args.eps);
    return y;
  }
};
template <class PTile>
struct Producer {
  using Sum = Sm90EVT<
      Sm90Compute<cutlass::plus, H, float, RN>,
      Sm90AccFetch,
      Sm90SrcFetch<H>>;
  using Reduce = Sm90EVT<
      Sm90ColReduction<
          cutlass::plus,
          cutlass::plus,
          cutlass::plus,
          0,
          PTile,
          float,
          float,
          RN,
          Stride<_1, _0, _0>,
          4,
          false,
          false>,
      Sm90EVT<Sm90Compute<Square, float, float, RN>, Sm90AccFetch>>;
  using Activated = Sm90EVT<
      Sm90Compute<cutlass::epilogue::thread::ReLu, H, float, RN>,
      Sm90EVT<
          Sm90Compute<cutlass::multiplies, float, float, RN>,
          Sm90AccFetch,
          Sm90RowBroadcast<0, PTile, H, float>>>;
  using Op = Sm90SplitTreeVisitor<Sum, Activated, Reduce>;
};
template <class CTile>
using Consumer = Sm90EVT<
    Sm90Compute<cutlass::multiply_add, H, float, RN>,
    Sm90AccFetch,
    Sm90EVT<
        Sm90Compute<Normalize, float, float, RN>,
        Sm90ColBroadcast<0, CTile, float, float>>,
    Sm90SrcFetch<H>>;
template <
    int ArchNumber,
    class T,
    bool Pingpong,
    class E,
    bool Independent = false>
struct Kernel {
  using Arch = conditional_t<
      ArchNumber == 100,
      cutlass::arch::Sm100,
      cutlass::arch::Sm90>;
  using Scheduler = conditional_t<
      ArchNumber == 100 || size<0>(T{}) >= 128 || Pingpong,
      ait::native_fusion::TailSchedulerTag,
      cutlass::gemm::PersistentScheduler>;
  using EpiSchedule = conditional_t<
      ArchNumber == 100,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      conditional_t<
          !Pingpong && size<0>(T{}) >= 128,
          cutlass::epilogue::TmaWarpSpecializedCooperative,
          cutlass::epilogue::TmaWarpSpecialized>>;
  using MainSchedule = conditional_t<
      ArchNumber == 100,
      cutlass::gemm::collective::KernelScheduleAuto,
      conditional_t<
          Pingpong,
          cutlass::gemm::KernelTmaWarpSpecializedPingpong,
          conditional_t<
              size<0>(T{}) >= 128,
              cutlass::gemm::KernelTmaWarpSpecializedCooperative,
              cutlass::gemm::KernelTmaWarpSpecialized>>>;

  using DefaultEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          Arch,
          cutlass::arch::OpClassTensorOp,
          T,
          Shape<_1, _1, _1>,
          cutlass::epilogue::collective::EpilogueTileAuto,
          float,
          float,
          H,
          L,
          8,
          H,
          L,
          8,
          EpiSchedule,
          E>::CollectiveOp;
  using Epilogue = conditional_t<
      Independent,
      typename ait::native_fusion::IndependentEpilogueStorage<
          DefaultEpilogue>::Type,
      DefaultEpilogue>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      H,
      L,
      8,
      H,
      cutlass::layout::ColumnMajor,
      8,
      float,
      T,
      Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
          typename Epilogue::SharedStorage)>,
      MainSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, Scheduler>;
  using G = cutlass::gemm::device::GemmUniversalAdapter<conditional_t<
      ArchNumber == 100,
      ait::native_fusion::StaticGemmKernel<GemmKernel>,
      conditional_t<!Pingpong && size<0>(T{}) == 64,
          ait::native_fusion::ParallelProducer90<GemmKernel>, GemmKernel>>>;
  static int run(
      void* a,
      void* b,
      void* c,
      void* d,
      int m,
      int n,
      int k,
      typename E::Arguments e,
      void* ws,
      cudaStream_t stream,
      int cluster = -1) {
    typename G::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {(H*)a,
         cutlass::make_cute_packed_stride(
             typename G::GemmKernel::StrideA{}, make_shape(m, k, 1)),
         (H*)b,
         cutlass::make_cute_packed_stride(
             typename G::GemmKernel::StrideB{}, make_shape(n, k, 1))},
        {e,
         (H*)c,
         cutlass::make_cute_packed_stride(
             typename G::GemmKernel::StrideC{}, make_shape(m, n, 1)),
         (H*)d,
         cutlass::make_cute_packed_stride(
             typename G::GemmKernel::StrideD{}, make_shape(m, n, 1))}};
    if (cudaGetDevice(&args.hw_info.device_id) != cudaSuccess ||
        cudaDeviceGetAttribute(
            &args.hw_info.sm_count,
            cudaDevAttrMultiProcessorCount,
            args.hw_info.device_id) != cudaSuccess)
      return int(cutlass::Status::kErrorInternal);
    args.scheduler.max_swizzle_size = 8;
    if (G::get_workspace_size(args) != 0)
      return int(cutlass::Status::kErrorWorkspaceNull);
    G gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess)
      return int(status);
    status = gemm.initialize(args, ws, stream);
    if (status != cutlass::Status::kSuccess)
      return int(status);
    // Clustered producer/boundary launches benefit this measured row range.
    // The SwiGLU consumer keeps its own cluster-launch threshold.
    if constexpr (ArchNumber == 90) {
      if (cluster > 0 || (cluster < 0 && m > 1792 && m <= 2560))
        return ait::native_fusion::launch_sm90_cluster(gemm, stream);
    }
    return int(gemm.run(stream, nullptr, true));
  }
};

template <
    int ArchNumber, int TileM, bool Pingpong, int ConsumerN,
    bool ProducerIndependent = (Pingpong || (ArchNumber == 100 && ConsumerN == 192)),
    bool ConsumerIndependent = (Pingpong || (ArchNumber == 100 && ConsumerN == 192))>
int run(
    void* x,
    void* w,
    void* skip,
    void* gamma,
    void* up,
    void* outer,
    void* out,
    uint8_t* workspace,
    int rows,
    float eps,
    cudaStream_t stream,
    int producer_cluster = -1,
    int consumer_cluster = -1) {
  using PT = Shape<Int<TileM>, Int<192>, _64>;
  using CT = Shape<_128, Int<ConsumerN>, _64>;
  using PS = Producer<PT>;
  using P = conditional_t<
      ArchNumber == 100,
      Sm90SplitTreeVisitor<
          typename PS::Sum,
          typename PS::Activated,
          Sm90EVT<ait::native_fusion::RowSquares, Sm90AccFetch>>,
      typename PS::Op>;
  using C = Consumer<CT>;
  void* tmp = workspace;
  float* sums = reinterpret_cast<float*>(
      workspace + ((int64_t(rows) * 384 + 255) / 256) * 256);
  typename P::Arguments p;
  if constexpr (ArchNumber == 100) {
    p = {{{}, {}, {}}, {{}, {sums}}, {{{}, {(H*)gamma, H(0), {}}, {}}, {}}};
  } else {
    p = {
        {{}, {}, {}},
        {{{}, {}}, {sums, 0.f, {}}},
        {{{}, {(H*)gamma, H(0), {}}, {}}, {}}};
  }
  int status = Kernel<ArchNumber, PT, Pingpong, P, ProducerIndependent>::run(
      x, w, skip, tmp, rows, 192, 576, p, nullptr, stream, producer_cluster);
  if (status)
    return status;
  typename C::Arguments c{{}, {{sums, 0.f, {}}, {eps}}, {}, {}};
  if constexpr (ArchNumber == 90 && TileM == 64 && !Pingpong && ConsumerN == 64) {
    if (rows <= 1296) {
      using SmallTile = Shape<_64, _64, _64>;
      using SmallConsumer = Consumer<SmallTile>;
      typename SmallConsumer::Arguments args{
          {}, {{sums, 0.f, {}}, {eps}}, {}, {}};
      return Kernel<90, SmallTile, false, SmallConsumer>::run(
          tmp, up, outer, out, rows, 384, 192, args, nullptr, stream, consumer_cluster);
    }
  }
  return Kernel<ArchNumber, CT, Pingpong, C, ConsumerIndependent>::run(
      tmp, up, outer, out, rows, 384, 192, c, nullptr, stream, consumer_cluster);
}

} // namespace ait::native_fusion::boundary
