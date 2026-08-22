#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
CuTeDSL (FlashAttention-4) backend for the ``flash_attention`` op.

Instead of emitting the vendored FMHA-v1 CUTLASS C++ kernel, this backend:
1. AOT-compiles FA4's forward via ``cute.compile()`` + ``export_to_c()`` --
   picking FA4's SM80 (Ampere) or SM90 (Hopper) forward by the target arch
   (see ``cutedsl_flash_attention_sm80.FlashAttentionFwdSm80Aot`` /
   ``cutedsl_flash_attention_sm90.FlashAttentionFwdSm90Aot``),
2. produces ``<func>_cutedsl.h`` + ``<func>_cutedsl.o`` (embedded cubin), and
3. returns a thin C++ wrapper (same AIT signature as the FMHA-v1 backend) that
   slices the packed QKV into strided Q/K/V views and launches the FA4 kernel.

The ``.o`` is linked into the model ``.so`` (build wiring keys on
``cutedsl_obj_path`` + ``-lcuda``), so the runtime path needs no Python.

Dispatch: ``ops.flash_attention`` selects this backend when the target is
created with ``use_cutedsl_attention=True``.

Scope: dense / equal-length sequences (the packed QKV is treated as
(batch, seq_len, 3, nheads, head_dim)); causal and head_dim are baked in at AOT
time.  Requires ``flash-attn-4`` + ``nvidia-cutlass-dsl`` (Python >= 3.10) in the
build env.
"""

import logging
import math
import os
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cuda.attention import flash_attention as _v1
from aitemplate.backend.target import Target

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# AOT compilation of the FA4 forward (SM80 / SM90)
# =============================================================================


def _aot_compile_cutedsl_kernel(output_dir, func_name, head_dim, is_causal, arch):
    """AOT-compile the FA4 forward for (head_dim, is_causal, arch); return (.h, .o).

    ``arch`` picks FA4's SM90 (Hopper) forward when >= 90, else the SM80 (Ampere)
    forward.  Both wrappers export the identical C interface
    ``(mQ, mK, mV, mO, mLSE, stream)``, so the generated C++ wrapper below is
    arch-agnostic.
    """
    import cuda.bindings.driver as cuda_drv
    import cutlass
    import torch

    # Compat shim: flash-attn's cute module (flash_attn/cute/flash_fwd.py) does
    # `import cutlass.utils.ampere_helpers as sm80_utils_basic` and reads
    # `SMEM_CAPACITY["sm80"]`. nvidia-cutlass-dsl 4.7.0 dropped that module (it
    # keeps hopper_helpers/blackwell_helpers only). The symbol is used solely on
    # FA4's SM80 path; we compile SM90 here, but flash_fwd imports it at module
    # load, so provide a minimal stand-in before importing flash_attn.cute. The
    # value mirrors cutlass_dsl SMEM_CAPACITY_MAP['sm_80'] (166912).
    import sys as _sys

    if "cutlass.utils.ampere_helpers" not in _sys.modules:
        import types as _types

        import cutlass.utils as _cu_utils

        if not hasattr(_cu_utils, "ampere_helpers"):
            _ah = _types.ModuleType("cutlass.utils.ampere_helpers")
            _ah.SMEM_CAPACITY = {"sm80": 166912}
            _sys.modules["cutlass.utils.ampere_helpers"] = _ah
            _cu_utils.ampere_helpers = _ah

    if arch >= 90:
        from aitemplate.backend.cuda.attention.cutedsl_flash_attention_sm90 import (
            FlashAttentionFwdSm90Aot as _FlashAttentionFwdAot,
        )
    else:
        from aitemplate.backend.cuda.attention.cutedsl_flash_attention_sm80 import (
            FlashAttentionFwdSm80Aot as _FlashAttentionFwdAot,
        )
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    # Representative dense tensors. batch/seqlen/nheads stay dynamic at runtime
    # (marked via to_cute_tensor's mark_layout_dynamic); head_dim is static.
    rep_b, rep_s, rep_h = 4, 256, 8
    d = head_dim
    q = torch.zeros(rep_b, rep_s, rep_h, d, device="cuda", dtype=torch.float16)
    k = torch.zeros(rep_b, rep_s, rep_h, d, device="cuda", dtype=torch.float16)
    v = torch.zeros(rep_b, rep_s, rep_h, d, device="cuda", dtype=torch.float16)
    o = torch.zeros(rep_b, rep_s, rep_h, d, device="cuda", dtype=torch.float16)
    lse = torch.zeros(rep_b, rep_h, rep_s, device="cuda", dtype=torch.float32)

    qt, kt, vt, ot = [to_cute_tensor(t, enable_tvm_ffi=False) for t in (q, k, v, o)]
    lset = to_cute_tensor(lse, assumed_align=4, enable_tvm_ffi=False)

    kernel = _FlashAttentionFwdAot(
        head_dim=d,
        softmax_scale=head_dim ** (-0.5),
        is_causal=is_causal,
        dtype=cutlass.Float16,
    )
    cu_stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    _LOGGER.info(
        f"CuTeDSL/FA4: AOT compiling flash_attention forward for {func_name} "
        f"(head_dim={d}, causal={is_causal}, SM{arch})"
    )
    import cutlass.cute as cute

    compiled = cute.compile(kernel, qt, kt, vt, ot, lset, cu_stream)

    os.makedirs(output_dir, exist_ok=True)
    cutedsl_name = f"{func_name}_cutedsl"
    compiled.export_to_c(file_path=output_dir, file_name=cutedsl_name)

    h_path = os.path.join(output_dir, f"{cutedsl_name}.h")
    o_path = os.path.join(output_dir, f"{cutedsl_name}.o")
    assert os.path.exists(h_path), f"CuTeDSL header not generated: {h_path}"
    assert os.path.exists(o_path), f"CuTeDSL object not generated: {o_path}"
    _LOGGER.info(
        f"CuTeDSL/FA4: AOT export done — {cutedsl_name}.h "
        f"({os.path.getsize(h_path)} bytes), {cutedsl_name}.o "
        f"({os.path.getsize(o_path)} bytes)"
    )
    return h_path, o_path


# =============================================================================
# C++ wrapper (same AIT signature as the FMHA-v1 backend)
# =============================================================================

CUTEDSL_WRAPPER_TEMPLATE = jinja2.Template(
    """
// Auto-generated CuTeDSL/FA4 wrapper for {{func_name}}
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "{{cutedsl_header}}"

namespace {

static {{func_name}}_cutedsl_Kernel_Module_t g_meta_{{func_name}};
static bool g_loaded_{{func_name}} = false;

static void ensure_cu_init_{{func_name}}() {
    static bool inited = false;
    if (!inited) {
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) {
            const char* e = nullptr;
            cuGetErrorString(r, &e);
            throw std::runtime_error(std::string("cuInit failed: ") + (e ? e : "?"));
        }
        inited = true;
    }
}

static void ensure_loaded_{{func_name}}() {
    if (!g_loaded_{{func_name}}) {
        ensure_cu_init_{{func_name}}();
        {{func_name}}_cutedsl_Kernel_Module_Load(&g_meta_{{func_name}});
        g_loaded_{{func_name}} = true;
    }
}

}  // namespace

{{func_signature}} {
    ensure_loaded_{{func_name}}();

    const int64_t B = batch_size;
    const int64_t S = seq_len;
    const int64_t Hh = num_heads;
    const int64_t D = head_size;

    const __half* qkv_h = reinterpret_cast<const __half*>(qkv);
    const int64_t qkv_row = 3 * Hh * D;  // stride between packed (b,s) rows

    // Q/K/V: strided views into packed qkv [B*S, 3, H, D] -> (B, S, H, D).
    {{func_name}}_cutedsl_Tensor_mQ_t tQ;
    tQ.data = (void*)(qkv_h + 0 * Hh * D);
    tQ.dynamic_shapes[0] = (int32_t)B; tQ.dynamic_shapes[1] = (int32_t)S;
    tQ.dynamic_shapes[2] = (int32_t)Hh; tQ.dynamic_shapes[3] = (int32_t)D;
    tQ.dynamic_strides[0] = S * qkv_row; tQ.dynamic_strides[1] = qkv_row;
    tQ.dynamic_strides[2] = D;

    {{func_name}}_cutedsl_Tensor_mK_t tK;
    tK.data = (void*)(qkv_h + 1 * Hh * D);
    tK.dynamic_shapes[0] = (int32_t)B; tK.dynamic_shapes[1] = (int32_t)S;
    tK.dynamic_shapes[2] = (int32_t)Hh; tK.dynamic_shapes[3] = (int32_t)D;
    tK.dynamic_strides[0] = S * qkv_row; tK.dynamic_strides[1] = qkv_row;
    tK.dynamic_strides[2] = D;

    {{func_name}}_cutedsl_Tensor_mV_t tV;
    tV.data = (void*)(qkv_h + 2 * Hh * D);
    tV.dynamic_shapes[0] = (int32_t)B; tV.dynamic_shapes[1] = (int32_t)S;
    tV.dynamic_shapes[2] = (int32_t)Hh; tV.dynamic_shapes[3] = (int32_t)D;
    tV.dynamic_strides[0] = S * qkv_row; tV.dynamic_strides[1] = qkv_row;
    tV.dynamic_strides[2] = D;

    // Output: contiguous [B*S, H, D] -> (B, S, H, D).
    {{func_name}}_cutedsl_Tensor_mO_t tO;
    tO.data = output;
    tO.dynamic_shapes[0] = (int32_t)B; tO.dynamic_shapes[1] = (int32_t)S;
    tO.dynamic_shapes[2] = (int32_t)Hh; tO.dynamic_shapes[3] = (int32_t)D;
    tO.dynamic_strides[0] = S * Hh * D; tO.dynamic_strides[1] = Hh * D;
    tO.dynamic_strides[2] = D;

    // LSE: contiguous (B, H, S) in the softmax_lse workspace.
    {{func_name}}_cutedsl_Tensor_mLSE_t tL;
    tL.data = softmax_lse;
    tL.dynamic_shapes[0] = (int32_t)B; tL.dynamic_shapes[1] = (int32_t)Hh;
    tL.dynamic_shapes[2] = (int32_t)S;
    tL.dynamic_strides[0] = Hh * S; tL.dynamic_strides[1] = S;

    (void)o_tmp; (void)p_dropout; (void)softmax_scale; (void)is_causal; (void)loop;
    (void)cu_seqlens;  // dense / equal-length only; scale + causal baked at AOT.

    cute_dsl_{{func_name}}_cutedsl_wrapper(
        &g_meta_{{func_name}}, &tQ, &tK, &tV, &tO, &tL, stream);
}
"""
)


# =============================================================================
# AIT backend registry functions
# =============================================================================


@registry.reg("cuda.flash_attention.gen_function_cutedsl")
def flash_attention_gen_function_cutedsl(func_attrs: Dict[str, Any]) -> str:
    current_target = Target.current()
    arch = int(current_target._arch)
    if arch < 80:
        raise NotImplementedError(
            f"FA4 CuTeDSL flash_attention requires SM80+, got SM{arch}"
        )

    workdir = func_attrs.get("workdir", "/tmp/ait_cutedsl")
    func_name = func_attrs["name"]
    head_dim = func_attrs["head_size"]
    is_causal = bool(func_attrs["causal"])

    _, o_path = _aot_compile_cutedsl_kernel(
        output_dir=workdir,
        func_name=func_name,
        head_dim=head_dim,
        is_causal=is_causal,
        arch=arch,
    )
    func_attrs["cutedsl_obj_path"] = o_path

    return CUTEDSL_WRAPPER_TEMPLATE.render(
        func_name=func_name,
        func_signature=_v1.FUNC_SIGNATURE.render(func_name=func_name),
        cutedsl_header=f"{func_name}_cutedsl.h",
    )


@registry.reg("cuda.flash_attention.func_decl_cutedsl")
def flash_attention_gen_function_decl_cutedsl(func_attrs: Dict[str, Any]):
    # Identical signature to the FMHA-v1 backend.
    return _v1.flash_attention_gen_function_decl(func_attrs)


@registry.reg("cuda.flash_attention.func_call_cutedsl")
def flash_attention_gen_function_call_cutedsl(func_attrs, indent="  "):
    """Same call site as the FMHA-v1 backend, but pass the ACTUAL seqlen.

    The FMHA-v1 backend pads ``seq_len`` up to a multiple of 256 and relies on
    ``cu_seqlens`` to mask the tail.  The dense FA4 wrapper instead reads the
    packed QKV as ``(batch, seq_len, 3, H, D)``, so it needs the *actual*
    per-sequence length (``max_seq_len``), not the padded one, or it would read
    past the real data.
    """
    x = func_attrs["inputs"][0]
    output_name = func_attrs["outputs"][0]._attrs["name"]
    qkv_name = x._attrs["name"]
    # cu_seqlens is optional: the dense/equal-length cutedsl path voids it, so callers may
    # omit the input -> pass nullptr (avoids a dummy constant in an inference graph).
    if len(func_attrs["inputs"]) > 1:
        seqlens_name = _v1.FUNC_CALL_INT32_PARAM_TEMPLATE.render(
            name=func_attrs["inputs"][1]._attrs["name"]
        )
    else:
        seqlens_name = "nullptr"

    batch_size = func_attrs["batch_size"]  # static max, for workspace/LSE sizing
    seq_len = func_attrs["max_seq_len"]  # actual length (not the 256-padded one)
    xshape = x._attrs["shape"]
    dense5d = len(xshape) == 5  # [B,S,3,H,D] vs packed 4D [total,3,H,D]
    num_heads = xshape[3 if dense5d else 2]._attrs["values"][0]
    head_size = xshape[4 if dense5d else 3]._attrs["values"][0]
    softmax_scale = head_size ** (-0.5)

    # Runtime batch for the kernel grid. Passing the baked max makes the kernel
    # process batch_size batches and write past the (runtime-sized) output for any
    # runtime B < batch_size, so a DYNAMIC batch must use the runtime dim variable.
    # 5D: dim0 IS B, use it directly. 4D packed: dim0 = total = B*seq_len, so
    # B = total / seq_len (exact). Workspace/LSE below keep the constant max, so
    # they never underflow at smaller runtime batch.
    dim0 = xshape[0]
    if len(dim0._attrs["values"]) > 1:  # dynamic batch
        batch_arg = (
            dim0._attrs["name"]
            if dense5d
            else "({} / {})".format(dim0._attrs["name"], seq_len)
        )
    else:
        batch_arg = str(dim0._attrs["values"][0] if dense5d else batch_size)

    return _v1.FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        qkv=qkv_name,
        cu_seqlens=seqlens_name,
        softmax_lse="reinterpret_cast<float*>(global_workspace_)",
        o_tmp="reinterpret_cast<float*>(global_workspace_ + {} * sizeof(float))".format(
            batch_size * num_heads * func_attrs["seq_len"]
        ),
        batch_size=batch_arg,
        seq_len=seq_len,
        num_heads=num_heads,
        head_size=head_size,
        p_dropout=func_attrs["dropout"],
        softmax_scale=softmax_scale,
        is_causal="true" if func_attrs["causal"] else "false",
        loop="true" if seq_len > 256 else "false",
        indent=indent,
    )


# =============================================================================
# q,k,v-separate FA4 op (flash_attention_qkv): O = FA4(Q,K,V) with Q,K,V,O each a
# contiguous [B,S,H,D] tensor -- NO packed-qkv concatenate. The AOT FA4 kernel already
# takes 3 separate tensor descriptors (compiled from separate contiguous q/k/v), so this
# only feeds them directly with contiguous strides. This is the fused-attention op backed
# by FA4 (the DotProductAttention / nvte_fused_attn role, cuDNN replaced by FA4).
# =============================================================================

FA4_QKV_SIGNATURE = jinja2.Template(
    "void {{func_name}}(void* q, void* k, void* v, void* o, int64_t B, "
    "uint8_t* workspace, cudaStream_t stream)"
)

FA4_QKV_WRAPPER_TEMPLATE = jinja2.Template(
    """
// Auto-generated CuTeDSL/FA4 q,k,v wrapper for {{func_name}}
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "{{cutedsl_header}}"

// Registry (defined in model_container.cu) drained by the Model ctor -> eager module load.
namespace ait { extern std::vector<void (*)()>& _cutedsl_loaders(); }

namespace {
static {{func_name}}_cutedsl_Kernel_Module_t g_meta_{{func_name}};
static bool g_loaded_{{func_name}} = false;
static void ensure_cu_init_{{func_name}}() {
    static bool inited = false;
    if (!inited) {
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) { const char* e = nullptr; cuGetErrorString(r, &e);
            throw std::runtime_error(std::string("cuInit failed: ") + (e ? e : "?")); }
        inited = true;
    }
}
static void ensure_loaded_{{func_name}}() {
    if (!g_loaded_{{func_name}}) {
        ensure_cu_init_{{func_name}}();
        {{func_name}}_cutedsl_Kernel_Module_Load(&g_meta_{{func_name}});
        g_loaded_{{func_name}} = true;
    }
}
// Register the loader at static init; the Model ctor drains the registry with a live
// context, so the module is loaded eagerly (before any graph capture).
struct {{func_name}}_reg_t {
    {{func_name}}_reg_t() { ait::_cutedsl_loaders().push_back(&ensure_loaded_{{func_name}}); }
};
static {{func_name}}_reg_t {{func_name}}_reg_inst;
}  // namespace

{{func_signature}} {
    ensure_loaded_{{func_name}}();  // no-op after eager load; kept for safety
    const int64_t S = {{s}}, Hh = {{h}}, D = {{d}};
    const int64_t row = Hh * D;  // contiguous [B,S,H,D]: stride between (b,s) rows

    {{func_name}}_cutedsl_Tensor_mQ_t tQ;
    tQ.data = q;
    tQ.dynamic_shapes[0] = (int32_t)B; tQ.dynamic_shapes[1] = (int32_t)S;
    tQ.dynamic_shapes[2] = (int32_t)Hh; tQ.dynamic_shapes[3] = (int32_t)D;
    tQ.dynamic_strides[0] = S * row; tQ.dynamic_strides[1] = row; tQ.dynamic_strides[2] = D;

    {{func_name}}_cutedsl_Tensor_mK_t tK;
    tK.data = k;
    tK.dynamic_shapes[0] = (int32_t)B; tK.dynamic_shapes[1] = (int32_t)S;
    tK.dynamic_shapes[2] = (int32_t)Hh; tK.dynamic_shapes[3] = (int32_t)D;
    tK.dynamic_strides[0] = S * row; tK.dynamic_strides[1] = row; tK.dynamic_strides[2] = D;

    {{func_name}}_cutedsl_Tensor_mV_t tV;
    tV.data = v;
    tV.dynamic_shapes[0] = (int32_t)B; tV.dynamic_shapes[1] = (int32_t)S;
    tV.dynamic_shapes[2] = (int32_t)Hh; tV.dynamic_shapes[3] = (int32_t)D;
    tV.dynamic_strides[0] = S * row; tV.dynamic_strides[1] = row; tV.dynamic_strides[2] = D;

    {{func_name}}_cutedsl_Tensor_mO_t tO;
    tO.data = o;
    tO.dynamic_shapes[0] = (int32_t)B; tO.dynamic_shapes[1] = (int32_t)S;
    tO.dynamic_shapes[2] = (int32_t)Hh; tO.dynamic_shapes[3] = (int32_t)D;
    tO.dynamic_strides[0] = S * row; tO.dynamic_strides[1] = row; tO.dynamic_strides[2] = D;

    {{func_name}}_cutedsl_Tensor_mLSE_t tL;
    tL.data = reinterpret_cast<float*>(workspace);   // [B,H,S] fp32 softmax-lse scratch
    tL.dynamic_shapes[0] = (int32_t)B; tL.dynamic_shapes[1] = (int32_t)Hh;
    tL.dynamic_shapes[2] = (int32_t)S;
    tL.dynamic_strides[0] = Hh * S; tL.dynamic_strides[1] = S;

    cute_dsl_{{func_name}}_cutedsl_wrapper(
        &g_meta_{{func_name}}, &tQ, &tK, &tV, &tO, &tL, stream);
}
"""
)

FA4_QKV_DECL_TEMPLATE = jinja2.Template("{{func_signature}};\n")

FA4_QKV_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{q}}, {{k}}, {{v}}, {{o}},
{{indent}}    {{b_expr}}, global_workspace_, stream
{{indent}});
"""
)


@registry.reg("cuda.flash_attention_qkv.gen_function")
def flash_attention_qkv_gen_function_cutedsl(func_attrs: Dict[str, Any]) -> str:
    current_target = Target.current()
    arch = int(current_target._arch)
    if arch < 80:
        raise NotImplementedError(
            f"FA4 CuTeDSL flash_attention_qkv requires SM80+, got SM{arch}"
        )
    workdir = func_attrs.get("workdir", "/tmp/ait_cutedsl")
    func_name = func_attrs["name"]
    _, o_path = _aot_compile_cutedsl_kernel(
        output_dir=workdir,
        func_name=func_name,
        head_dim=func_attrs["head_dim"],
        is_causal=bool(func_attrs["causal"]),
        arch=arch,
    )
    func_attrs["cutedsl_obj_path"] = o_path
    sig = FA4_QKV_SIGNATURE.render(func_name=func_name)
    return FA4_QKV_WRAPPER_TEMPLATE.render(
        func_name=func_name,
        func_signature=sig,
        cutedsl_header=f"{func_name}_cutedsl.h",
        s=func_attrs["seq_len"],
        h=func_attrs["heads"],
        d=func_attrs["head_dim"],
    )


@registry.reg("cuda.flash_attention_qkv.func_decl")
def flash_attention_qkv_gen_function_decl(func_attrs: Dict[str, Any]):
    return FA4_QKV_DECL_TEMPLATE.render(
        func_signature=FA4_QKV_SIGNATURE.render(func_name=func_attrs["name"])
    )


@registry.reg("cuda.flash_attention_qkv.func_call")
def flash_attention_qkv_gen_function_call(func_attrs, indent="  "):
    q, k, v = func_attrs["inputs"]
    o = func_attrs["outputs"][0]
    return FA4_QKV_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        q=q._attrs["name"],
        k=k._attrs["name"],
        v=v._attrs["name"],
        o=o._attrs["name"],
        b_expr=q._attrs["shape"][0]._attrs["name"],
    )
