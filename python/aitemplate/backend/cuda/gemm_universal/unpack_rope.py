#  Backend codegen for `unpack_rope`: reads the packed qkv gemm output [rows, 3*dim]
#  (dim = H*D, laid out [.,3,H,D]), RoPE-rotates the q and k heads, copies v, and writes
#  three CONTIGUOUS [rows, H, D] tensors. One read-once/write-once pass replaces the 3-way
#  split + two rope2d kernels. head_dim==16 uses an all-128-bit thread-per-head kernel;
#  other head dims fall back to a scalar per-head kernel.
#
#  rmsnorm-prologue: when an rrms [rows,1] input is given, each token's q/k/v is multiplied
#  by rrms[token] here (the gemm ran on RAW x with gamma folded into its weight), so the
#  separate RMSNorm kernel's [rows,C] write of the normalized activation is eliminated.
import jinja2

from aitemplate.backend import registry

# head_dim == 16 (P==8): x head = 2 uint4, cos/sin = 1 uint4, register-resident.
VEC16_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
// gid = token*H + h. packed q head @ base, k @ base+HD, v @ base+2*HD (HD = dim = H*D).
__global__ void {{func_name}}_kernel(const __half* __restrict__ qkv,
                                     const __half* __restrict__ cosb,
                                     const __half* __restrict__ sinb,
                                     __half* __restrict__ q, __half* __restrict__ k,
                                     __half* __restrict__ v,
{% if has_rrms %}                                     const __half* __restrict__ rrms,
{% endif %}                                     long long rowsH) {
  const int H = {{h}}, HD = {{hd}}, SH = {{sh}};
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= rowsH) return;
  const long long token = gid / H;
  const int h = (int)(gid - token * H);
{% if has_rrms %}  const float rf = __half2float(rrms[token]);
{% else %}  const float rf = 1.0f;
{% endif %}  const __half* base = qkv + token * (3LL * HD) + h * 16;  // q head (D=16)
  const long long cbase = (long long)(gid % SH) * 8;
  const uint4 cq = reinterpret_cast<const uint4*>(cosb + cbase)[0];
  const uint4 sq = reinterpret_cast<const uint4*>(sinb + cbase)[0];
  const __half* ch = reinterpret_cast<const __half*>(&cq);
  const __half* sh = reinterpret_cast<const __half*>(&sq);
  // q, k: (rrms-scale then) rope, in registers
#pragma unroll
  for (int which = 0; which < 2; which++) {
    const __half* src = base + which * HD;              // q then k
    __half* dst = (which == 0 ? q : k) + gid * 16LL;
    uint4 a = reinterpret_cast<const uint4*>(src)[0];
    uint4 b = reinterpret_cast<const uint4*>(src)[1];
    __half2* h0 = reinterpret_cast<__half2*>(&a);
    __half2* h1 = reinterpret_cast<__half2*>(&b);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 p = __half22float2(h0[j]);
      p.x *= rf; p.y *= rf;
      float c = __half2float(ch[j]), s = __half2float(sh[j]);
      h0[j] = __float22half2_rn(make_float2(p.x * c - p.y * s, p.x * s + p.y * c));
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 p = __half22float2(h1[j]);
      p.x *= rf; p.y *= rf;
      float c = __half2float(ch[4 + j]), s = __half2float(sh[4 + j]);
      h1[j] = __float22half2_rn(make_float2(p.x * c - p.y * s, p.x * s + p.y * c));
    }
    reinterpret_cast<uint4*>(dst)[0] = a;
    reinterpret_cast<uint4*>(dst)[1] = b;
  }
  // v: (rrms-scale then) copy
  const __half* vb = base + 2 * HD;
{% if has_rrms %}
  uint4 v0 = reinterpret_cast<const uint4*>(vb)[0];
  uint4 v1 = reinterpret_cast<const uint4*>(vb)[1];
  __half2* vh0 = reinterpret_cast<__half2*>(&v0);
  __half2* vh1 = reinterpret_cast<__half2*>(&v1);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float2 p = __half22float2(vh0[j]); vh0[j] = __float22half2_rn(make_float2(p.x * rf, p.y * rf));
    float2 qf = __half22float2(vh1[j]); vh1[j] = __float22half2_rn(make_float2(qf.x * rf, qf.y * rf));
  }
  reinterpret_cast<uint4*>(v + gid * 16LL)[0] = v0;
  reinterpret_cast<uint4*>(v + gid * 16LL)[1] = v1;
{% else %}
  reinterpret_cast<uint4*>(v + gid * 16LL)[0] = reinterpret_cast<const uint4*>(vb)[0];
  reinterpret_cast<uint4*>(v + gid * 16LL)[1] = reinterpret_cast<const uint4*>(vb)[1];
{% endif %}
}
}  // namespace

void {{func_name}}(const void* qkv, const void* cosb, const void* sinb, void* q, void* k,
                   void* v, {% if has_rrms %}const void* rrms, {% endif %}int64_t rows,
                   cudaStream_t stream) {
  const long long rowsH = (long long)rows * {{h}};
  const int block = 256;
  const unsigned int grid = (unsigned int)((rowsH + block - 1) / block);
  {{func_name}}_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(qkv), reinterpret_cast<const __half*>(cosb),
      reinterpret_cast<const __half*>(sinb), reinterpret_cast<__half*>(q),
      reinterpret_cast<__half*>(k), reinterpret_cast<__half*>(v),
{% if has_rrms %}      reinterpret_cast<const __half*>(rrms),
{% endif %}      rowsH);
}
"""
)

# general: one thread per head, scalar over P pairs / D copy.
SCALAR_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
__global__ void {{func_name}}_kernel(const __half* __restrict__ qkv,
                                     const __half* __restrict__ cosb,
                                     const __half* __restrict__ sinb,
                                     __half* __restrict__ q, __half* __restrict__ k,
                                     __half* __restrict__ v,
{% if has_rrms %}                                     const __half* __restrict__ rrms,
{% endif %}                                     long long rowsH) {
  const int H = {{h}}, HD = {{hd}}, SH = {{sh}}, D = {{d}}, P = {{p}};
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= rowsH) return;
  const long long token = gid / H;
  const int h = (int)(gid - token * H);
{% if has_rrms %}  const float rf = __half2float(rrms[token]);
{% else %}  const float rf = 1.0f;
{% endif %}  const __half* base = qkv + token * (3LL * HD) + h * D;
  const long long cb = (long long)(gid % SH) * P;
  for (int which = 0; which < 2; which++) {  // q, k
    const __half* src = base + which * HD;
    __half* dst = (which == 0 ? q : k) + gid * (long long)D;
    for (int p = 0; p < P; p++) {
      float x0 = __half2float(src[2 * p]) * rf, x1 = __half2float(src[2 * p + 1]) * rf;
      float c = __half2float(cosb[cb + p]), s = __half2float(sinb[cb + p]);
      dst[2 * p] = __float2half(x0 * c - x1 * s);
      dst[2 * p + 1] = __float2half(x0 * s + x1 * c);
    }
  }
  const __half* vb = base + 2 * HD;
  __half* vd = v + gid * (long long)D;
  for (int d = 0; d < D; d++) vd[d] = __float2half(__half2float(vb[d]) * rf);
}
}  // namespace

void {{func_name}}(const void* qkv, const void* cosb, const void* sinb, void* q, void* k,
                   void* v, {% if has_rrms %}const void* rrms, {% endif %}int64_t rows,
                   cudaStream_t stream) {
  const long long rowsH = (long long)rows * {{h}};
  const int block = 256;
  const unsigned int grid = (unsigned int)((rowsH + block - 1) / block);
  {{func_name}}_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(qkv), reinterpret_cast<const __half*>(cosb),
      reinterpret_cast<const __half*>(sinb), reinterpret_cast<__half*>(q),
      reinterpret_cast<__half*>(k), reinterpret_cast<__half*>(v),
{% if has_rrms %}      reinterpret_cast<const __half*>(rrms),
{% endif %}      rows * {{h}});
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    "\nvoid {{func_name}}(const void*, const void*, const void*, void*, void*, void*, "
    "{% if has_rrms %}const void*, {% endif %}int64_t, cudaStream_t);\n"
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{qkv}}, {{cos}}, {{sin}}, {{q}}, {{k}}, {{v}}, {% if rrms %}{{rrms}}, {% endif %}{{rows_expr}}, stream
{{indent}});
"""
)


@registry.reg("cuda.unpack_rope.gen_function")
def unpack_rope_gen_function(func_attrs):
    has_rrms = func_attrs.get("has_rrms", False)
    tmpl = VEC16_TEMPLATE if func_attrs["head_dim"] == 16 else SCALAR_TEMPLATE
    return tmpl.render(
        func_name=func_attrs["name"],
        h=func_attrs["heads"],
        hd=func_attrs["dim"],
        sh=func_attrs["sh"],
        d=func_attrs["head_dim"],
        p=func_attrs["p"],
        has_rrms=has_rrms,
    )


@registry.reg("cuda.unpack_rope.func_decl")
def unpack_rope_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"], has_rrms=func_attrs.get("has_rrms", False)
    )


@registry.reg("cuda.unpack_rope.func_call")
def unpack_rope_gen_function_call(func_attrs, indent="  "):
    has_rrms = func_attrs.get("has_rrms", False)
    qkv, cosb, sinb = func_attrs["inputs"][:3]
    rrms = func_attrs["inputs"][3]._attrs["name"] if has_rrms else None
    q, k, v = func_attrs["outputs"]
    qshape = qkv._attrs["shape"]  # [B, S, 3*dim]
    rows_expr = " * ".join(d._attrs["name"] for d in qshape[:-1]) or "1"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        qkv=qkv._attrs["name"],
        cos=cosb._attrs["name"],
        sin=sinb._attrs["name"],
        q=q._attrs["name"],
        k=k._attrs["name"],
        v=v._attrs["name"],
        rrms=rrms,
        rows_expr=rows_expr,
    )
