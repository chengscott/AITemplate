// Gated-epilogue design adapted from QuACK 0.6.5.
// Copyright (c) 2025-2026, QuACK team.
// Reimplemented for native CUTLASS C++; see licenses/LICENSE.quack.txt.
#pragma once
#include "native_rms_relu_gemm.h"

#include <cuda_fp16.h>
#include "cute/arch/copy_sm100.hpp"
#include "native_concat_mainloop_sm100.h"
#include "native_concat_mainloop_sm90.h"
#include "native_direct_epilogue.h"
#include "native_packed_math.h"
#include "native_swiglu_epilogue_sm90.h"
#include "native_early_norm_sm90.h"
namespace ait::native_fusion::swiglu {
using namespace cute;
using namespace cutlass::epilogue::fusion;
using H = cutlass::half_t;
using L = cutlass::layout::RowMajor;
// Gate and up weights occupy separate contiguous halves of the input matrix.
// Paired TMA loads place both halves in one accumulator tile.
template <int TileN>
struct Activation : Sm90VisitorImpl<> {
  static_assert(TileN % 64 == 0 && 1152 % TileN == 0);
  struct SharedStorage {};
  struct Arguments {
    float const* sums;
    H* out;
    float eps;
  };
  using Params = Arguments;
  template <class P>
  static bool can_implement(P const&, Arguments const&) {
    return true;
  }
  template <class P>
  static size_t get_workspace_size(P const&, Arguments const&) {
    return 0;
  }
  template <class P>
  static auto initialize_workspace(
      P const&,
      Arguments const&,
      void*,
      cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  template <class P>
  static Params to_underlying_arguments(P const&, Arguments const& a, void*) {
    return a;
  }
  CUTLASS_HOST_DEVICE Activation(Params const&, SharedStorage const&) {}
  template <class CT>
  struct Callbacks : EmptyConsumerStoreCallbacks {
    CT coord;
    Params p;
    int rows;
    float normalization;
    float gate[TileN / 2];
    cutlass::AlignedArray<H, TileN / 2, 16> result;
    CUTLASS_DEVICE Callbacks(CT c, Params a, int m) : coord(c), p(a), rows(m) {
      // The caller has waited for input dependencies. Fetch the row scale
      // while MMA is in flight for the latency-sensitive narrow tile.
      if constexpr (TileN == 64) {
        int row = get<0>(coord(_, _, _, 0, 0)(0));
        normalization =
            row < rows ? float(H(rsqrtf(p.sums[row] / 192.f + p.eps))) : 0.f;
      }
    }
    template <class A, int F>
    CUTLASS_DEVICE cutlass::Array<H, F> visit(
        cutlass::Array<A, F> const& acc,
        int ev,
        int em,
        int en) {
      auto coords = coord(_, _, _, em, en);
      int row = get<0>(coords(ev * F)), col = get<1>(coords(ev * F));
      cutlass::Array<H, F> dummy;
      dummy.clear();
      if (row >= rows)
        return dummy;
      float r;
      if constexpr (TileN == 64)
        r = normalization;
      else
        r = float(H(rsqrtf(p.sums[row] / 192.f + p.eps)));
      // Compile-time subtile indexing keeps the gate values in registers.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < F; i += 2) {
        constexpr int half = TileN / 2;
        int n = en * F + i;
        if (n < half) {
          gate[n] = float(acc[i]);
          gate[n + 1] = float(acc[i + 1]);
        } else {
          auto rr = make_float2(r, r);
          auto g =
              pair_mul(make_float2(gate[n - half], gate[n - half + 1]), rr);
          auto u = pair_mul(make_float2(float(acc[i]), float(acc[i + 1])), rr);
          auto e = pair_mul(
              g, make_float2(-1.4426950408889634f, -1.4426950408889634f));
          auto d = pair_add(
              make_float2(approx_exp2(e.x), approx_exp2(e.y)),
              make_float2(1.f, 1.f));
          auto z = pair_mul(
              pair_mul(g, make_float2(approx_rcp(d.x), approx_rcp(d.y))), u);
          reinterpret_cast<__half2*>(&result)[(n - half) / 2] =
              __floats2half2_rn(z.x, z.y);
        }
      }
      if ((en + 1) * F == TileN) {
        // Each store fills a complete 32-byte sector despite the row-per-lane
        // accumulator layout. Narrower stores duplicate traffic into L2.
        auto* dst = reinterpret_cast<cute::uint256_t*>(
            p.out + int64_t(row) * 576 + (col / TileN) * (TileN / 2));
        auto* src = reinterpret_cast<uint32_t*>(&result);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < TileN / 32; ++i) {
          int j = i * 8;
          cute::SM100_STORE_256bit_CACHE_NOALLOCATION::copy(
              src[j],
              src[j + 1],
              src[j + 2],
              src[j + 3],
              src[j + 4],
              src[j + 5],
              src[j + 6],
              src[j + 7],
              dst[i]);
        }
      }
      return dummy;
    }
  };
};
template <
    int N,
    int ArchNumber = 100,
    int TM = 128,
    bool Pingpong = true,
    bool EarlyNorm = false>
struct Ffn {
  static_assert(ArchNumber == 90 || ArchNumber == 100);
  static_assert(!EarlyNorm || (ArchNumber == 90 && TM == 128 && Pingpong));
  static_assert(N == 64 || N == 128 || N == 192);
  static_assert(TM == 64 || TM == 128);
  using Arch = conditional_t<
      ArchNumber == 100,
      cutlass::arch::Sm100,
      cutlass::arch::Sm90>;
  using Schedule = conditional_t<
      ArchNumber == 100,
      cutlass::gemm::collective::KernelScheduleAuto,
      conditional_t<
          Pingpong,
          cutlass::gemm::KernelTmaWarpSpecializedPingpong,
          cutlass::gemm::KernelTmaWarpSpecialized>>;
  using T = Shape<Int<TM>, Int<N>, _64>;
  using HalfT = Shape<Int<TM>, Int<N / 2>, _64>;
  using Cluster = Shape<_1, _1, _1>;
  using BaseEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      T,
      Cluster,
      conditional_t<
          ArchNumber == 100,
          Shape<_128, _64>,
          cutlass::epilogue::collective::EpilogueTileAuto>,
      float,
      float,
      void,
      L,
      8,
      H,
      L,
      8,
      conditional_t<
          ArchNumber == 100,
          cutlass::epilogue::TmaWarpSpecialized1Sm,
          cutlass::epilogue::TmaWarpSpecialized>,
      Activation<N>>::CollectiveOp;
  using Epi = conditional_t<
      ArchNumber == 100,
      ait::native_fusion::DirectEpilogue<BaseEpi>,
      Direct90<BaseEpi, N, EarlyNorm>>;
  template <class Tile, class Stages>
  using Builder = cutlass::gemm::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      H,
      L,
      8,
      H,
      cutlass::layout::ColumnMajor,
      8,
      float,
      Tile,
      Cluster,
      Stages,
      Schedule>;
  using Full = typename Builder<
      T,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
          typename Epi::SharedStorage)>>::CollectiveOp;
  using Half = typename Builder<
      HalfT,
      cutlass::gemm::collective::StageCount<Full::DispatchPolicy::Stages>>::
      CollectiveOp;
  using Main = conditional_t<
      ArchNumber == 100,
      ait::native_fusion::ConcatMainloop<Full, Half>,
      ait::native_fusion::ConcatMainloop90<Full, Half>>;
  using Base = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Main,
      Epi,
      conditional_t<
          ArchNumber == 100 || Pingpong,
          ait::native_fusion::TailSchedulerTag,
          cutlass::gemm::PersistentScheduler>>;
  using G = cutlass::gemm::device::GemmUniversalAdapter<conditional_t<
      ArchNumber == 100,
      ait::native_fusion::StaticGemmKernel<Base>,
      conditional_t<
          EarlyNorm, ait::native_fusion::EarlyNormKernel90<Base>, Base>>>;
  static int run(
      void* x,
      void* w,
      void* r,
      void* out,
      int m,
      float eps,
      cudaStream_t stream,
      int cluster = -1) {
    typename Epi::Arguments epi{};
    epi.thread = {(float*)r, (H*)out, eps};
    epi.ptr_D = (H*)out;
    epi.dC = cutlass::make_cute_packed_stride(
        typename Base::StrideC{}, make_shape(m, 1152, 1));
    epi.dD = epi.dC;
    typename G::Arguments a{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, 1152, 192, 1},
        {(H*)x,
         cutlass::make_cute_packed_stride(
             typename Base::StrideA{}, make_shape(m, 192, 1)),
         (H*)w,
         cutlass::make_cute_packed_stride(
             typename Base::StrideB{}, make_shape(1152, 192, 1))},
        epi};
    if (cudaGetDevice(&a.hw_info.device_id) != cudaSuccess ||
        cudaDeviceGetAttribute(
            &a.hw_info.sm_count,
            cudaDevAttrMultiProcessorCount,
            a.hw_info.device_id) != cudaSuccess)
      return int(cutlass::Status::kErrorInternal);
    a.scheduler.max_swizzle_size = 8;
    if (G::get_workspace_size(a) != 0)
      return int(cutlass::Status::kErrorWorkspaceNull);
    G g;
    auto s = g.can_implement(a);
    if (s != cutlass::Status::kSuccess)
      return int(s);
    s = g.initialize(a, nullptr, stream);
    if (s != cutlass::Status::kSuccess)
      return int(s);
    if constexpr (ArchNumber == 90) {
      if (cluster > 0 || (cluster < 0 && m > 1792 && m <= 2592))
        return ait::native_fusion::launch_sm90_cluster(g, stream);
    }
    return int(g.run(stream, nullptr, true));
  }
};
template <class T>
struct Identity {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const {
    return x;
  }
};
template <int ArchNumber, int TM, bool Independent = false>
int residual(
    void* x,
    void* proj,
    void* skip,
    void* updated,
    float* sums,
    int rows,
    cudaStream_t stream,
    int cluster = -1) {
  using Tile = Shape<Int<TM>, Int<192>, _64>;
  using P = boundary::Producer<Tile>;
  using Pass =
      Sm90EVT<Sm90Compute<Identity, H, float, boundary::RN>, Sm90AccFetch>;
  if constexpr (ArchNumber == 100) {
    using E = Sm90EVT<RowSquares, typename P::Sum>;
    typename E::Arguments e{{{}, {}, {}}, {sums}};
    return boundary::Kernel<ArchNumber, Tile, false, E, Independent>::run(
        x, proj, skip, updated, rows, 192, 192, e, nullptr, stream, cluster);
  } else {
    using E = Sm90SplitTreeVisitor<typename P::Sum, Pass, typename P::Reduce>;
    typename E::Arguments e{
        {{}, {}, {}}, {{{}, {}}, {sums, 0.f, {}}}, {{}, {}}};
    return boundary::Kernel<ArchNumber, Tile, false, E, Independent>::run(
        x, proj, skip, updated, rows, 192, 192, e, nullptr, stream, cluster);
  }
}
static __global__ void sum_squares(H const* x, float* out, int rows) {
  int64_t row = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (row >= rows)
    return;
  float ss = 0;
  if (lane < 24) {
    auto v = reinterpret_cast<cutlass::AlignedArray<H, 8, 16> const*>(
        x + int64_t(row) * 192)[lane];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      float f = float(v[i]);
      ss += f * f;
    }
  }
  CUTLASS_PRAGMA_UNROLL
  for (int d = 16; d; d >>= 1)
    ss += __shfl_xor_sync(0xffffffff, ss, d);
  if (lane == 0)
    out[row] = ss;
}
template <int ArchNumber, bool Joint>
int prepare(
    void*& x,
    void* w,
    void* skip,
    void* proj,
    void* updated,
    void* out,
    uint8_t* workspace,
    int rows,
    float eps,
    cudaStream_t stream,
    int producer_cluster = -1) {
  auto* sums = reinterpret_cast<float*>(workspace);
  if constexpr (Joint) {
    int status;
    if constexpr (ArchNumber == 90) {
      if (rows <= 5184)
        status = residual<90, 64>(x, proj, skip, updated, sums, rows, stream, producer_cluster);
      else
        status = residual<90, 128>(x, proj, skip, updated, sums, rows, stream, producer_cluster);
    } else {
      if (rows <= 10368)
        status = residual<100, 128>(x, proj, skip, updated, sums, rows, stream, producer_cluster);
      else
        status = residual<100, 128, true>(
            x, proj, skip, updated, sums, rows, stream, producer_cluster);
    }
    if (status)
      return status;
    x = updated;
  } else {
    sum_squares<<<(int64_t(rows) + 3) / 4, 128, 0, stream>>>((H*)x, sums, rows);
    if (cudaGetLastError() != cudaSuccess)
      return int(cutlass::Status::kErrorInternal);
  }
  return 0;
}
template <int ArchNumber, bool Joint, int TileM, int TileN, bool Pingpong, bool EarlyNorm>
int run_tuned(
    void* x,
    void* w,
    void* skip,
    void* proj,
    void* updated,
    void* out,
    uint8_t* workspace,
    int rows,
    float eps,
    cudaStream_t stream,
    int producer_cluster,
    int consumer_cluster) {
  int status = prepare<ArchNumber, Joint>(x, w, skip, proj, updated, out, workspace, rows, eps, stream, producer_cluster);
  if (status) return status;
  auto* sums = reinterpret_cast<float*>(workspace);
  return Ffn<TileN, ArchNumber, TileM, Pingpong, EarlyNorm>::run(x, w, sums, out, rows, eps, stream, consumer_cluster);
}
template <int ArchNumber, bool Joint>
int run(
    void* x,
    void* w,
    void* skip,
    void* proj,
    void* updated,
    void* out,
    uint8_t* workspace,
    int rows,
    float eps,
    cudaStream_t stream) {
  constexpr int producer_cluster = -1;
  int status = prepare<ArchNumber, Joint>(x, w, skip, proj, updated, out, workspace, rows, eps, stream, producer_cluster);
  if (status) return status;
  auto* sums = reinterpret_cast<float*>(workspace);
  if constexpr (ArchNumber == 100) {
    if (rows <= 2592)
      return Ffn<64>::run(x, w, sums, out, rows, eps, stream);
    return Ffn<128>::run(x, w, sums, out, rows, eps, stream);
  } else {
    if (rows <= 324)
      return Ffn<64, 90, 64, false>::run(x, w, sums, out, rows, eps, stream);
    if (rows <= 648)
      return Ffn<128, 90, 64, false>::run(x, w, sums, out, rows, eps, stream);
    if (rows <= 1296)
      return Ffn<192, 90, 64, false>::run(x, w, sums, out, rows, eps, stream);
    // Narrow output tiles improve CTA coverage at intermediate row counts.
    if (rows <= 1792)
      return Ffn<64, 90, 128, true, true>::run(
          x, w, sums, out, rows, eps, stream);
    if (rows <= 2592)
      return Ffn<128, 90, 128, true, true>::run(
          x, w, sums, out, rows, eps, stream);
    return Ffn<192, 90, 128, true>::run(x, w, sums, out, rows, eps, stream);
  }
}
} // namespace ait::native_fusion::swiglu
