#pragma once
namespace ait::native_fusion {
// SM100 packed FP32 arithmetic keeps two activation values in each instruction.
static __device__ __forceinline__ float2 pair_mul(float2 a, float2 b) {
  union Bits {
    float2 f;
    unsigned long long u;
  };
  Bits x{a}, y{b}, z;
  asm("mul.f32x2 %0, %1, %2;" : "=l"(z.u) : "l"(x.u), "l"(y.u));
  return z.f;
}
static __device__ __forceinline__ float2 pair_add(float2 a, float2 b) {
  union Bits {
    float2 f;
    unsigned long long u;
  };
  Bits x{a}, y{b}, z;
  asm("add.f32x2 %0, %1, %2;" : "=l"(z.u) : "l"(x.u), "l"(y.u));
  return z.f;
}
static __device__ __forceinline__ float2 half_pair(float x, float y) {
  return __half22float2(__floats2half2_rn(x, y));
}
static __device__ __forceinline__ float approx_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
static __device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

} // namespace ait::native_fusion
