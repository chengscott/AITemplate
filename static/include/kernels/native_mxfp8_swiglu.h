#pragma once
#include "native_gdc.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "cute/arch/copy_sm100.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "native_direct_epilogue.h"
#include "native_mxfp8_mainloop.h"
#include "native_packed_math.h"
#include "native_sm100_pipeline.h"
#include "native_tail_scheduler.h"
namespace ait::native_fusion::mxfp8 {
using namespace cute;
using namespace cutlass::epilogue::fusion;
using H = cutlass::half_t;
using L = cutlass::layout::RowMajor;
// The input weight rows alternate gate/up. A CTA owns complete 32-value output
// blocks.
template <int TileN>
struct QuantizedSwiGLU : Sm90VisitorImpl<> {
  using ElementAux = H;
  struct SharedStorage {};
  struct Arguments {
    unsigned char* out;
    unsigned char* scales;
    H const* rms;
  };
  using Params = Arguments;
  template <class PS>
  static bool can_implement(PS const&, Arguments const&) {
    return true;
  }
  template <class PS>
  static size_t get_workspace_size(PS const&, Arguments const&) {
    return 0;
  }
  template <class PS>
  static cutlass::Status initialize_workspace(
      PS const&,
      Arguments const&,
      void*,
      cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  Params params;
  template <class PS>
  static Params to_underlying_arguments(PS const&, Arguments const& a, void*) {
    return a;
  }
  CUTLASS_HOST_DEVICE QuantizedSwiGLU(Params const& p, SharedStorage const& s)
      : params(p) {}
  template <class CT>
  struct Callbacks : EmptyConsumerStoreCallbacks {
    CT coords;
    Params p;
    int rows;
    float normalization;
    CUTLASS_DEVICE Callbacks(CT c, Params p_, int r)
        : coords(c), p(p_), rows(r) {
      // The kernel waits for dependencies before constructing callbacks.
      // Hide the scale load behind MMA for narrow tiles only.
      if constexpr (TileN <= 128) {
        int row = get<0>(coords(_, _, _, 0, 0)(0));
        normalization = row < rows ? float(p.rms[row]) : 0.f;
      }
    }
    template <class A, int F>
    CUTLASS_DEVICE cutlass::Array<H, F> visit(
        cutlass::Array<A, F> const& acc,
        int ev,
        int em,
        int en) {
      static_assert(
          F == 64,
          "Each thread must own one complete output quantization block");
      auto coord = coords(_, _, _, em, en);
      int row = get<0>(coord(ev * F)), col = get<1>(coord(ev * F)) / 2;
      cutlass::Array<H, F> result;
      result.clear();
      if (row >= rows || col >= 576)
        return result;
      float r;
      if constexpr (TileN <= 128)
        r = normalization;
      else
        r = float(p.rms[row]);
      float2 rr = make_float2(r, r);
      float2 v[16];
      float maxima[16];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 16; ++i) {
        float2 g = pair_mul(half_pair(acc[4 * i], acc[4 * i + 2]), rr);
        float2 u = pair_mul(half_pair(acc[4 * i + 1], acc[4 * i + 3]), rr);
        float2 ex = pair_mul(
            g, make_float2(-1.4426950408889634f, -1.4426950408889634f));
        float2 den = pair_add(
            make_float2(approx_exp2(ex.x), approx_exp2(ex.y)),
            make_float2(1.f, 1.f));
        float2 z = pair_mul(
            pair_mul(g, make_float2(approx_rcp(den.x), approx_rcp(den.y))), u);
        v[i] = half_pair(z.x, z.y);
        maxima[i] = fmaxf(fabsf(v[i].x), fabsf(v[i].y));
      }
      CUTLASS_PRAGMA_UNROLL
      for (int stride = 8; stride > 0; stride /= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < stride; ++i)
          maxima[i] = fmaxf(maxima[i], maxima[i + stride]);
      }
      float amax = maxima[0];
      cutlass::float_ue8m0_t scale(amax * (1.f / 448.f));
      int exponent = scale.storage;
      int kb = col / 32, iy = row % 128;
      // The last consumer K tile reads two padded scale slots as well.
      if (col == 512) {
        p.scales
            [int64_t(row / 128) * 2560 + 4 * 512 + (iy % 32) * 16 +
             (iy / 32) * 4 + 2] = 0;
        p.scales
            [int64_t(row / 128) * 2560 + 4 * 512 + (iy % 32) * 16 +
             (iy / 32) * 4 + 3] = 0;
      }
      p.scales
          [int64_t(row / 128) * 2560 + (kb / 4) * 512 + (iy % 32) * 16 +
           (iy / 32) * 4 + kb % 4] = exponent;
      float inv = amax > 0 ? __uint_as_float((254 - exponent) << 23) : 0.f;
      uint4 packed[2];
      auto* pairs = reinterpret_cast<unsigned short*>(packed);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 16; ++i)
        pairs[i] = __nv_fp8x2_e4m3(pair_mul(v[i], make_float2(inv, inv))).__x;
      auto* dst = reinterpret_cast<cute::uint256_t*>(
          p.out + int64_t(row) * 576 + col);
      // Fill each output sector once instead of issuing two partial stores.
      cute::SM100_STORE_256bit_CACHE_NOALLOCATION::copy(
          packed[0].x,
          packed[0].y,
          packed[0].z,
          packed[0].w,
          packed[1].x,
          packed[1].y,
          packed[1].z,
          packed[1].w,
          *dst);
      return result;
    }
  };
  template <bool Ref, class... A>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      ConsumerStoreArgs<A...> const& args) {
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto [M, N, K, B] = args.problem_shape_mnkl;
    auto id = make_identity_tensor(make_shape(M, N));
    auto tile =
        local_tile(id, take<0, 2>(args.tile_shape_mnk), make_coord(m, n));
    auto coords = sm90_partition_for_epilogue<Ref>(
        tile, args.epi_tile, args.tiled_copy, args.thread_idx);
    return Callbacks<decltype(coords)>(coords, params, M);
  }
};

using MX = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
template <bool Producer, int TileN>
struct Kernel {
  using PTile = Shape<_128, Int<TileN>, _128>;
  using T = PTile;
  using Fusion = conditional_t<
      Producer,
      QuantizedSwiGLU<TileN>,
      cutlass::epilogue::fusion::LinearCombination<H, float, H, float>>;
  using EC = conditional_t<Producer, void, H>;
  using ED = H;
  using BaseEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100,
          cutlass::arch::OpClassBlockScaledTensorOp,
          T,
          Shape<_1, _1, _1>,
          conditional_t<
              Producer,
              Shape<_128, _64>,
              cutlass::epilogue::collective::EpilogueTileAuto>,
          float,
          float,
          EC,
          L,
          8,
          ED,
          L,
          8,
          cutlass::epilogue::TmaWarpSpecialized1Sm,
          Fusion>::CollectiveOp;
  using Epilogue =
      conditional_t<Producer, DirectEpilogue<BaseEpilogue>, BaseEpilogue>;
  using OriginalMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100,
          cutlass::arch::OpClassBlockScaledTensorOp,
          MX,
          L,
          16,
          MX,
          cutlass::layout::ColumnMajor,
          16,
          float,
          T,
          Shape<_1, _1, _1>,
          cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
              typename Epilogue::SharedStorage)>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Mainloop = ait::native_fusion::BlockScaledMainloop<OriginalMainloop>;
  using GK = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Mainloop,
      Epilogue,
      ait::native_fusion::TailSchedulerTag>;
  using G = cutlass::gemm::device::GemmUniversalAdapter<
      ait::native_fusion::StaticGemmKernel<GK>>;
  static int run(
      void* a,
      void* b,
      void* sa,
      void* sb,
      void* c,
      void* d,
      void* sh,
      void* r,
      int m,
      void* ws,
      cudaStream_t stream) {
    constexpr int n = Producer ? 1152 : 192, k = Producer ? 192 : 576;
    auto shape = make_shape(m, n, k, 1);
    using Config = typename Mainloop::Sm1xxBlkScaledConfig;
    typename Epilogue::Arguments epi{};
    if constexpr (Producer) {
      epi.thread = {(unsigned char*)d, (unsigned char*)sh, (H*)r};
      epi.ptr_D = (H*)d;
    } else {
      epi.thread.alpha = 1;
      epi.thread.beta = 1;
      epi.ptr_C = (H*)c;
      epi.ptr_D = (H*)d;
    }
    epi.dC = cutlass::make_cute_packed_stride(
        typename GK::StrideC{}, make_shape(m, n, 1));
    epi.dD = cutlass::make_cute_packed_stride(
        typename GK::StrideD{}, make_shape(m, n, 1));
    typename G::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        shape,
        {(typename MX::DataType*)a,
         cutlass::make_cute_packed_stride(
             typename GK::StrideA{}, make_shape(m, k, 1)),
         (typename MX::DataType*)b,
         cutlass::make_cute_packed_stride(
             typename GK::StrideB{}, make_shape(n, k, 1)),
         (typename MX::ScaleFactorType*)sa,
         Config::tile_atom_to_shape_SFA(shape),
         (typename MX::ScaleFactorType*)sb,
         Config::tile_atom_to_shape_SFB(shape)},
        epi};
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
    return int(gemm.run(stream, nullptr, true));
  }
};

static __global__ void prepare_quant(
    const __half* __restrict__ a,
    __half* __restrict__ rrms_out,
    float eps,
    unsigned char* __restrict__ aq,
    unsigned char* __restrict__ sfa,
    long long rows) {
  constexpr int nkb = 6, ntx = 2;

  const int warps_per_cta = blockDim.x / 8;
  const long long row =
      (long long)blockIdx.x * warps_per_cta + (threadIdx.x / 8);
  if (row >= rows)
    return;
  const int lane = threadIdx.x % 8;
  const long long K = (long long)nkb * 32;
  const uint4* ar = reinterpret_cast<const uint4*>(a + row * K);
  uint2* aqr = reinterpret_cast<uint2*>(aq + row * K);
  const long long iy = row & 127;
  if (lane >= 6)
    sfa[(row >> 7) * 1024 + 512 + (iy % 32) * 16 + (iy >> 5) * 4 + (lane & 3)] =
        0;

  uint4 xbuf[1][4];
  float ss = 0.f;
  int bi = 0;
#pragma unroll
  for (int kb = lane; kb < nkb; kb += 8, bi++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      uint4 q = ar[kb * 4 + j];
      xbuf[bi][j] = q;
      const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f = __half22float2(h[i]);
        ss += f.x * f.x + f.y * f.y;
      }
    }
  }
#pragma unroll
  for (int o = 4; o > 0; o >>= 1)
    ss += __shfl_xor_sync(__activemask(), ss, o);
  const float rrms = rsqrtf(ss / (float)K + eps);
  if (lane == 0)
    rrms_out[row] = __float2half(rrms);
  bi = 0;
#pragma unroll
  for (int kb = lane; kb < nkb; kb += 8, bi++) {
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const __half2* h = reinterpret_cast<const __half2*>(&xbuf[bi][j]);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f = __half22float2(h[i]);
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
    int b = amax > 0.f ? ((int)ceilf(log2f(amax * (1.0f / 448.0f))) + 127) : 0;
    b = b < 0 ? 0 : (b > 254 ? 254 : b);
    float inv = amax > 0.f ? exp2f((float)(127 - b)) : 0.f;
    sfa[(row >> 7) * (long long)ntx * 512 + (kb >> 2) * 512 + (iy % 32) * 16 +
        (iy >> 5) * 4 + (kb & 3)] = (unsigned char)b;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const __half2* h = reinterpret_cast<const __half2*>(&xbuf[bi][j]);
      uint2 out;
      unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f = __half22float2(h[i]);
        f.x *= inv;
        f.y *= inv;
        os[i] = __nv_fp8x2_e4m3(f).__x;
      }
      aqr[kb * 4 + j] = out;
    }
  }
}

template <int TileN, int DownN = 0>
int run(
    void* x,
    void* w,
    void* sf,
    void* down,
    void* dsf,
    void* out,
    uint8_t* workspace,
    int m,
    float eps,
    cudaStream_t stream) {
  auto aligned = [](int64_t n) { return (n + 255) / 256 * 256; };
  int64_t rm = (m + 127) / 128;
  auto* aq = workspace;
  auto* sa = aq + aligned(m * 192ll);
  auto* r = sa + rm * 1024;
  auto* h = r + aligned(m * 2ll);
  auto* sh = h + aligned(m * 576ll);
  prepare_quant<<<(m + 31) / 32, 256, 0, stream>>>(
      (half*)x, (half*)r, eps, aq, sa, m);
  if (cudaGetLastError() != cudaSuccess)
    return int(cutlass::Status::kErrorInternal);
  int status = Kernel<true, TileN>::run(
      aq, w, sa, sf, nullptr, h, sh, r, m, nullptr, stream);
  if (status)
    return status;
  if constexpr (DownN != 0) {
    return Kernel<false, DownN>::run(
        h, down, sh, dsf, x, out, nullptr, nullptr, m, nullptr, stream);
  }
  // Narrow down-projection tiles reduce latency when the grid is small.
  if (m <= 10368)
    return Kernel<false, 64>::run(
        h, down, sh, dsf, x, out, nullptr, nullptr, m, nullptr, stream);
  return Kernel<false, 192>::run(
      h, down, sh, dsf, x, out, nullptr, nullptr, m, nullptr, stream);
}

} // namespace ait::native_fusion::mxfp8
