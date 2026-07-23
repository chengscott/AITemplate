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
1. AOT-compiles FA4's SM80 forward via ``cute.compile()`` + ``export_to_c()``
   (see ``cutedsl_flash_attention_sm80.FlashAttentionFwdSm80Aot``),
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
# AOT compilation of the FA4 SM80 forward
# =============================================================================


def _aot_compile_cutedsl_kernel(output_dir, func_name, head_dim, is_causal):
    """AOT-compile FA4 SM80 forward for (head_dim, is_causal); return (.h, .o)."""
    import cuda.bindings.driver as cuda_drv
    import cutlass
    import torch

    from aitemplate.backend.cuda.attention.cutedsl_flash_attention_sm80 import (
        FlashAttentionFwdSm80Aot,
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

    kernel = FlashAttentionFwdSm80Aot(
        head_dim=d,
        softmax_scale=head_dim ** (-0.5),
        is_causal=is_causal,
        dtype=cutlass.Float16,
    )
    cu_stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    _LOGGER.info(
        f"CuTeDSL/FA4: AOT compiling flash_attention forward for {func_name} "
        f"(head_dim={d}, causal={is_causal})"
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
    seqlens_name = _v1.FUNC_CALL_INT32_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][1]._attrs["name"]
    )

    batch_size = func_attrs["batch_size"]
    seq_len = func_attrs["max_seq_len"]  # actual length (not the 256-padded one)
    num_heads = x._attrs["shape"][2]._attrs["values"][0]
    head_size = x._attrs["shape"][3]._attrs["values"][0]
    softmax_scale = head_size ** (-0.5)

    return _v1.FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        qkv=qkv_name,
        cu_seqlens=seqlens_name,
        softmax_lse="reinterpret_cast<float*>(global_workspace_)",
        o_tmp="reinterpret_cast<float*>(global_workspace_ + {} * sizeof(float))".format(
            batch_size * num_heads * func_attrs["seq_len"]
        ),
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_size=head_size,
        p_dropout=func_attrs["dropout"],
        softmax_scale=softmax_scale,
        is_causal="true" if func_attrs["causal"] else "false",
        loop="true" if seq_len > 256 else "false",
        indent=indent,
    )
