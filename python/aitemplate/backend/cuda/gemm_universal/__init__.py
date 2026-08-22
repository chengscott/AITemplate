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
# flake8: noqa
from aitemplate.backend.cuda.gemm_universal import (
    bmm_rcr_permute,
    bmm_rrr_permute,
    bmm_xxx,
    bmm_xxx_add,
    cutedsl_bmm,
    gemm_rcr_bias,
    gemm_rcr_bias_activation_cutedsl,
    gemm_rcr_bias_cutedsl,
    gemm_rcr_bias_elementwise,
    gemm_rcr_bias_fast_gelu,
    gemm_rcr_bias_gelu,
    gemm_rcr_bias_hardswish,
    gemm_rcr_bias_permute,
    gemm_rcr_bias_relu,
    gemm_rcr_bias_sigmoid,
    gemm_rcr_bias_swish,
    gemm_rcr_bias_tanh,
    gemm_rcr_cutedsl,
    gemm_rcr_fast_gelu,
    gemm_rcr_permute,
    gemm_rcr_permute_elup1,
    gemm_rrr,
    gemm_rrr_bias,
    gemm_rrr_permute,
    group_gemm_rcr,
    group_gemm_rcr_bias,
    group_gemm_rcr_bias_relu,
    group_gemm_rcr_bias_sigmoid,
    rmsnorm,
    rms_reduce,
    rope2d,
    swiglu,
    unpack_rope,
    quantize_to_fp8,
    dequant_fp8,
    im2col_fp8,
    conv2d_fp8,
    gemm_rcr_fp8_fused,
    quantize_to_fp8_tensor,
    quantize_to_fp8_amax,
    perm021fc_ccr,
    perm021fc_ccr_bias,
    perm021fc_ccr_bias_permute,
    perm021fc_crc,
    perm021fc_crc_bias,
    perm102_bmm_rcr,
    perm102_bmm_rcr_bias,
    perm102_bmm_rrr,
    perm102_bmm_rrr_bias,
)
