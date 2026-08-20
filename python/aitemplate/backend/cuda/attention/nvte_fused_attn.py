#  Backend codegen for nvte_fused_attn: wraps TransformerEngine's nvte_fused_attn_fwd
#  (BSHD, no-mask, fp16 inference) with its two-pass workspace + aux-tensor-pack query.
#  All scratch (cu_seqlens, rng, softmax-stats aux, nvte workspace) is carved out of
#  global_workspace_ at fixed max offsets; cu_seqlens = [0,S,..,B*S] filled by a kernel.
import jinja2

from aitemplate.backend import registry

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_runtime.h>
#include <vector>
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/fused_attn.h"

using transformer_engine::TensorWrapper;
using transformer_engine::DType;

namespace {
__global__ void {{func_name}}_fill_cu(int* cq, int* ck, int bp1, int S) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < bp1) { cq[i] = i * S; ck[i] = i * S; }
}
__host__ __device__ inline size_t {{func_name}}_dbytes(int t) {
  // NVTEDType: Float32=3? use runtime sizes: kNVTEFloat32=... map via values
  switch (t) {
    case kNVTEFloat32: return 4; case kNVTEFloat16: case kNVTEBFloat16: return 2;
    case kNVTEByte: return 1; case kNVTEInt32: return 4; case kNVTEInt64: return 8;
    default: return 4;
  }
}
}  // namespace

// O[B,S,H,D] = Attention(Q,K,V) (BSHD, no mask, scale). fp16.
void {{func_name}}(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr,
                   int64_t B, uint8_t* ws, cudaStream_t stream) {
  const int S = {{s}}, H = {{h}}, D = {{d}};
  const float scale = {{scale}}f;
  const int64_t maxB = {{maxb}};

  // ---- carve global_workspace_ (fixed max offsets) ----
  int* cu_q = reinterpret_cast<int*>(ws);
  int* cu_kv = cu_q + (maxB + 1);
  size_t off = (size_t)2 * (maxB + 1) * sizeof(int);
  off = (off + 7) & ~size_t(7);
  long long* rng = reinterpret_cast<long long*>(ws + off);
  off += 2 * sizeof(long long);
  off = (off + 15) & ~size_t(15);
  uint8_t* aux_region = ws + off;
  size_t aux_cap = (size_t)maxB * H * S * sizeof(float) + 512;
  uint8_t* nvte_ws_region = aux_region + aux_cap;

  int bp1 = (int)(B + 1);
  {{func_name}}_fill_cu<<<(bp1 + 63) / 64, 64, 0, stream>>>(cu_q, cu_kv, bp1, S);
  cudaMemsetAsync(rng, 0, 2 * sizeof(long long), stream);

  std::vector<size_t> qs{(size_t)B, (size_t)S, (size_t)H, (size_t)D};
  TensorWrapper Q(q_ptr, qs, DType::kFloat16), K(k_ptr, qs, DType::kFloat16),
      V(v_ptr, qs, DType::kFloat16), O(o_ptr, qs, DType::kFloat16);
  TensorWrapper Bias, SoftmaxOffset, Sd, cuqp, cukvp, ptk, ptv;
  TensorWrapper CUQ(cu_q, std::vector<size_t>{(size_t)(B + 1)}, DType::kInt32);
  TensorWrapper CUKV(cu_kv, std::vector<size_t>{(size_t)(B + 1)}, DType::kInt32);
  TensorWrapper RNG(rng, std::vector<size_t>{2}, DType::kInt64);

  NVTETensorPack pack;
  nvte_tensor_pack_create(&pack);
  TensorWrapper wsq;  // pass 1: empty -> query workspace + aux shapes

#define {{func_name}}_CALL(WS) \
  nvte_fused_attn_fwd(Q.data(), K.data(), V.data(), Bias.data(), SoftmaxOffset.data(), \
      Sd.data(), O.data(), &pack, CUQ.data(), CUKV.data(), cuqp.data(), cukvp.data(), \
      ptk.data(), ptv.data(), RNG.data(), (size_t)S, (size_t)S, /*is_training=*/false, \
      /*return_max_logit=*/false, /*cuda_graph=*/false, scale, /*dropout=*/0.0f, \
      NVTE_BSHD_BSHD_BSHD, NVTE_BSHD, NVTE_BSHD, NVTE_NO_BIAS, NVTE_NO_MASK, \
      NVTE_VANILLA_SOFTMAX, /*wl=*/-1, /*wr=*/-1, /*bottom_right=*/false, (WS).data(), stream)

  {{func_name}}_CALL(wsq);

  // place aux tensors into the reserved aux_region
  uint8_t* ap = aux_region;
  for (size_t i = 0; i < pack.size; ++i) {
    NVTEShape sh = nvte_tensor_shape(pack.tensors[i]);
    NVTEDType dt = nvte_tensor_type(pack.tensors[i]);
    size_t n = 1;
    for (size_t j = 0; j < sh.ndim; ++j) n *= sh.data[j];
    NVTEBasicTensor bt{ap, dt, sh};
    nvte_set_tensor_param_v2(pack.tensors[i], kNVTERowwiseData, &bt, sizeof(bt));
    size_t b = n * {{func_name}}_dbytes((int)dt);
    ap += (b + 15) & ~size_t(15);
  }
  // nvte workspace from its reserved region
  NVTEShape wshape = wsq.shape();
  std::vector<size_t> wsv;
  for (size_t i = 0; i < wshape.ndim; ++i) wsv.push_back(wshape.data[i]);
  TensorWrapper W(nvte_ws_region, wsv, wsq.dtype());

  {{func_name}}_CALL(W);  // pass 2: compute
#undef {{func_name}}_CALL
  nvte_tensor_pack_destroy(&pack);
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(void*, void*, void*, void*, int64_t, uint8_t*, cudaStream_t);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{q_ptr}}, {{k_ptr}}, {{v_ptr}}, {{o_ptr}},
{{indent}}    {{b_expr}}, global_workspace_, stream
{{indent}});
"""
)


@registry.reg("cuda.nvte_fused_attn.gen_function")
def nvte_fused_attn_gen_function(func_attrs):
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        s=func_attrs["seq_len"],
        h=func_attrs["heads"],
        d=func_attrs["head_dim"],
        scale=func_attrs["scale"],
        maxb=func_attrs["max_batch"],
    )


@registry.reg("cuda.nvte_fused_attn.func_decl")
def nvte_fused_attn_gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.nvte_fused_attn.func_call")
def nvte_fused_attn_gen_function_call(func_attrs, indent="  "):
    q, k, v = func_attrs["inputs"]
    o = func_attrs["outputs"][0]
    b_expr = q._attrs["shape"][0]._attrs["name"]
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        q_ptr=q._attrs["name"],
        k_ptr=k._attrs["name"],
        v_ptr=v._attrs["name"],
        o_ptr=o._attrs["name"],
        b_expr=b_expr,
    )
