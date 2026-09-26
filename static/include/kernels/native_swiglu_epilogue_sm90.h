// Gated-epilogue design adapted from QuACK 0.6.5.
// Copyright (c) 2025-2026, QuACK team.
// Reimplemented for native CUTLASS C++; see licenses/LICENSE.quack.txt.
#pragma once
#include "native_rms_relu_gemm.h"
#include "native_direct_epilogue.h"
#include <cuda_fp16.h>
namespace ait::native_fusion::swiglu {
template <class B, int N, bool EarlyNorm = false>
struct Direct90 : ait::native_fusion::DirectEpilogue<B> {
  using Parent = ait::native_fusion::DirectEpilogue<B>;
  using typename Parent::LoadPipeline;
  using typename Parent::LoadPipelineState;
  using typename Parent::Params;
  using typename Parent::StorePipeline;
  using typename Parent::StorePipelineState;
  using typename Parent::TensorStorage;
  static constexpr bool RequiresTransactionBytes = false;
  CUTLASS_DEVICE Direct90(Params const& p, TensorStorage& s) : Parent(p, s) {}
  float cached_norm[4];

  template <class PS, class T, class C, class MMA>
  CUTLASS_DEVICE void prefetch_norm(PS problem, T tile, C coord, MMA mma, int tid) {
    using namespace cute;
    auto id = local_tile(
        make_identity_tensor(make_shape(get<0>(problem), get<1>(problem))),
        take<0, 2>(tile), make_coord(get<0>(coord), get<1>(coord)));
    auto cs = coalesce(mma.get_slice(tid).partition_C(id));
    static_assert(size(cs) / (N / 4) == 4);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      int row = get<0>(cs((j / 2) * (N / 2) + (j % 2) * 2));
      float sum = 0.f;
      if (row < get<0>(problem)) {
        // Keep this load ahead of MMA instead of sinking it into the epilogue.
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(sum)
            : "l"(this->direct_params.sums + row) : "memory");
        cached_norm[j] = float(cutlass::half_t(
            rsqrtf(sum / 192.f + this->direct_params.eps)));
      } else {
        cached_norm[j] = 0.f;
      }
    }
  }
  template <class T>
  static constexpr int get_store_pipe_increment(T) {
    return 0;
  }
  template <class PS, class T, class C, class AE, class AL, class MMA>
  CUTLASS_DEVICE auto store(
      LoadPipeline,
      LoadPipelineState ls,
      StorePipeline,
      StorePipelineState ss,
      PS problem,
      T tile,
      C coord,
      cute::Tensor<AE, AL> acc,
      MMA mma,
      int tid,
      TensorStorage&,
      int subtile = -1) {
    using namespace cute;
    auto [M, NN, K, L] = problem;
    auto id = local_tile(
        make_identity_tensor(make_shape(M, NN)),
        take<0, 2>(tile),
        make_coord(get<0>(coord), get<1>(coord)));
    auto cs = coalesce(mma.get_slice(tid).partition_C(id));
    auto values = coalesce(acc);
    auto p = this->direct_params;
    // Load each row's normalization once, before any potentially aliasing
    // stores.
    float rms[size(values) / (N / 4)];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size(values) / (N / 4); ++j) {
      int row = get<0>(cs((j / 2) * (N / 2) + (j % 2) * 2));
      if constexpr (EarlyNorm) {
        rms[j] = cached_norm[j];
      } else {
        rms[j] = row < M
            ? float(cutlass::half_t(rsqrtf(p.sums[row] / 192.f + p.eps)))
            : 0.f;
      }
    }
    // Redistribute two eight-column fragments across each four-lane group.
    // Four adjacent 8-byte stores then fill one 32-byte sector.
    // Keep shuffles outside the row predicate so all lanes participate.
    int quad_lane = threadIdx.x % 4;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(values); i += 8) {
      if (i % (N / 2) >= N / 4) {
        CUTLASS_PRAGMA_UNROLL
        for (int ro = 0; ro < 4; ro += 2) {
          int j = i + ro;
          auto c = cs(j);
          int row = get<0>(c), col = get<1>(c);
          float r = rms[2 * (j / (N / 2)) + (j % 4) / 2];
          float g0 = values(j - N / 4) * r, g1 = values(j - N / 4 + 1) * r;
          float g2 = values(j + 4 - N / 4) * r, g3 = values(j + 5 - N / 4) * r;
          __half2 lop = __floats2half2_rn(
              g0 / (1.f + __expf(-g0)) * (values(j) * r),
              g1 / (1.f + __expf(-g1)) * (values(j + 1) * r));
          __half2 hip = __floats2half2_rn(
              g2 / (1.f + __expf(-g2)) * (values(j + 4) * r),
              g3 / (1.f + __expf(-g3)) * (values(j + 5) * r));
          uint32_t lo = *reinterpret_cast<uint32_t*>(&lop);
          uint32_t hi = *reinterpret_cast<uint32_t*>(&hip);
          int source = 2 * (quad_lane % 2);
          uint32_t a0 = __shfl_sync(0xffffffff, lo, source, 4);
          uint32_t a1 = __shfl_sync(0xffffffff, lo, source + 1, 4);
          uint32_t b0 = __shfl_sync(0xffffffff, hi, source, 4);
          uint32_t b1 = __shfl_sync(0xffffffff, hi, source + 1, 4);
          uint2 packed =
              quad_lane < 2 ? make_uint2(a0, a1) : make_uint2(b0, b1);
          if (row < M) {
            auto* dst = p.out + int64_t(row) * 576 + col -
                get<1>(coord) * (N / 2) - N / 2 + 2 * quad_lane;
            *reinterpret_cast<uint2*>(dst) = packed;
          }
        }
      }
    }
    return make_tuple(ls, ss);
  }
  CUTLASS_DEVICE auto store_tail(
      LoadPipeline,
      LoadPipelineState ls,
      StorePipeline,
      StorePipelineState ss) {
    return cute::make_tuple(ls, ss);
  }
};
} // namespace ait::native_fusion::swiglu
