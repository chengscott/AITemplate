#pragma once

#include <cuda_fp16.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "native_dual_gemm_sm80.h"

namespace ait::native_fusion::sm80 {
using H = cutlass::half_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;

// Each warp owns a complete row; no cross-CTA reduction or atomic reset.
static __global__ void inverse_rms(H const* input, float* inv, int rows, float eps) {
  int row = blockIdx.x * 4 + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (row >= rows) return;
  float sum = 0.f;
  #pragma unroll
  for (int col = lane; col < 192; col += 32) {
    float x = float(input[int64_t(row) * 192 + col]);
    sum += x * x;
  }
  #pragma unroll
  for (int offset = 16; offset; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if (!lane) inv[row] = float(H(rsqrtf(sum / 192.f + eps)));
}

template<bool InlineRms, int Count>
struct RmsSwiGLU {
  using ElementOutput = H;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int kCount = Count;
  using FragmentOutput = cutlass::Array<H, kCount>;
  using FragmentAccumulator = cutlass::Array<float, kCount>;
  struct Params { float const* inv; int rows; H const* input; float eps; };
  Params params;
  CUTLASS_HOST_DEVICE RmsSwiGLU(Params const& p) : params(p) {}
  CUTLASS_DEVICE FragmentOutput operator()(
      FragmentAccumulator const& gate, FragmentAccumulator const& up, int row, int row_lanes) const {
    FragmentOutput result;
    float scale;
    if constexpr (InlineRms) {
      float sum = 0.f;
      if(row < params.rows) {
        for(int col=threadIdx.x % row_lanes;col<192;col+=row_lanes) {
          float v=float(params.input[int64_t(row)*192+col]); sum+=v*v;
        }
      }
      for(int d=row_lanes/2;d;d/=2) sum+=__shfl_xor_sync(0xffffffff,sum,d,row_lanes);
      scale=float(H(rsqrtf(sum/192.f+params.eps)));
    } else scale = row < params.rows ? params.inv[row] : 0.f;
    #pragma unroll
    for (int i = 0; i < kCount; ++i) {
      float g = gate[i] * scale;
      float u = up[i] * scale;
      result[i] = H((g / (1.f + expf(-g))) * u);
    }
    return result;
  }
};

template<int M, int N, int Stages, bool InlineRms=false>
int swiglu(void const* x, void const* weight, void* out,
           float* inv, int rows, float eps, cudaStream_t stream) {
  constexpr int Access = M < 64 ? N/16 : 8;
  using Linear = cutlass::epilogue::thread::LinearCombination<H,Access,float,float>;
  using Gemm = cutlass::gemm::device::NativeDualGemm<
      H, Row, H, Col, Col, H, Row, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<M,N,32>,
      cutlass::gemm::GemmShape<(M < 64 ? 32 : M/2),N/2,32>,
      cutlass::gemm::GemmShape<16,8,16>,
      Linear, Linear, RmsSwiGLU<InlineRms,Access>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages, false, false>;
  if (rows <= 0) return 0;
  if constexpr (!InlineRms)
    inverse_rms<<<(rows-1)/4+1,128,0,stream>>>(static_cast<H const*>(x),inv,rows,eps);
  if (cudaGetLastError() != cudaSuccess) return int(cutlass::Status::kErrorInternal);
  typename Gemm::Arguments args{
      cutlass::gemm::DualGemmMode::kGemm, {rows,576,192},
      {static_cast<H const*>(x),192}, {static_cast<H const*>(weight),192},
      {nullptr,576}, {nullptr,576},
      {static_cast<H const*>(weight)+576*192,192},
      {nullptr,576}, {nullptr,576}, {static_cast<H*>(out),576},
      {1.f,0.f}, {1.f,0.f}, {inv,rows,static_cast<H const*>(x),eps}};
  Gemm gemm;
  auto status = gemm.can_implement(args);
  if (status == cutlass::Status::kSuccess) status = gemm(args,nullptr,stream);
  return int(status);
}

template<int M=64, int N=64, int Stages=3>
int residual(void const* x, void const* w, void const* skip,
             void* out, int rows, int k, cudaStream_t stream) {
  using Gemm = cutlass::gemm::device::Gemm<
      H,Row,H,Col,H,Row,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<M,N,32>,cutlass::gemm::GemmShape<M/2,N/2,32>,
      cutlass::gemm::GemmShape<16,8,16>,
      cutlass::epilogue::thread::LinearCombination<H,8,float,float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,Stages>;
  typename Gemm::Arguments args{{rows,192,k},
      {static_cast<H const*>(x),k},{static_cast<H const*>(w),k},
      {static_cast<H const*>(skip),192},{static_cast<H*>(out),192},{1.f,1.f}};
  Gemm gemm;
  return int(gemm(args,nullptr,stream));
}

template<int M=64, int N=64, int Stages=3, bool Joint=true, bool InlineRms=false>
int run(void const* x,void const* w,void const* skip,void const* proj,
        void* updated,void* out,uint8_t* workspace,int rows,float eps,cudaStream_t stream) {
  if (rows <= 0) return 0;
  if constexpr (Joint) {
    int status = residual(x,proj,skip,updated,rows,192,stream);
    if (status) return status;
    x = updated;
  }
  return swiglu<M,N,Stages,InlineRms>(x,w,out,reinterpret_cast<float*>(workspace),rows,eps,stream);
}
} // namespace ait::native_fusion::sm80
