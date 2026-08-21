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
Util functions for CUDA codegen.
"""

import logging

from collections import OrderedDict

from aitemplate.backend import registry
from aitemplate.utils.mk_cutlass_lib.mk_cutlass_lib import mk_cutlass_lib

# pylint: disable=C0103,C0415,W0707


_LOGGER = logging.getLogger(__name__)


class Args:
    def __init__(self, arch):
        self.operations = "all"
        self.build_dir = ""
        self.curr_build_dir = ""
        self.generator_target = ""
        self.architectures = arch
        self.kernels = "all"
        self.ignore_kernels = ""
        # cutlass >= 3.2 Manifest reads these additional fields
        self.exclude_kernels = ""
        self.instantiation_level = ""
        self.cuda_version = "11.4.0"
        self.kernel_filter_file = None
        self.selected_kernel_list = None
        self.interface_dir = None
        self.filter_by_cc = True
        self.disable_full_archs_compilation = False


registry.reg("cuda.make_cutlass_lib")(mk_cutlass_lib)


@registry.reg("cuda.gen_cutlass_ops")
def gen_ops(
    arch,
    cuda_version,
    allow_cutlass_sm90,
    force_cutlass_sm90,
):
    import cutlass_lib

    args = Args(arch)
    if cuda_version is not None:
        args.cuda_version = cuda_version
    manifest = cutlass_lib.manifest.Manifest(args)

    if arch == "90":
        if force_cutlass_sm90:
            cutlass_lib.generator.GenerateSM90(manifest, args.cuda_version)
            # SM80 fallback ops accompany the forced SM90 pool for GEMMs only.
            # This fork's CUTLASS 3.x (Hopper) host codegen supports TMA warp-
            # specialized (align-8) epilogues: gemms whose output N is not a
            # multiple of 8 (policy/value heads N=1/2, ...) can't realize an
            # SM90 3.x TMA kernel and fall back to SM80 gemm kernels (which run
            # correctly on Hopper). The align-8 gemms still get and (via
            # profiling) prefer their SM90 3.x TMA kernels.
            #
            # Convolutions are DIFFERENT: this fork now has a native SM90 3.x
            # conv host path (backend/cuda/conv2d/common.py: emit_instance_3x /
            # EXEC_TEMPLATE_3X, with hand-authored bias/relu/residual fusion
            # epilogues). To force the profiler to pick those native SM90 conv
            # kernels (not an SM80 fallback), we DROP every SM80 (2.x) conv op
            # from the manifest below, keeping only the is_3x conv ops that
            # GenerateSM90 produced. GEMM ops (SM80 + SM90) are untouched.
            cutlass_lib.generator.GenerateSM80(manifest, args.cuda_version)
            cutlass_lib.extra_operation.GenerateSM80(manifest, args)
            _generate_sm90_conv3x_f16_f32acc(cutlass_lib, manifest)
            # Opt-in (NBT_AIT_FP8_CONV=1): also emit native SM90 3.x fprop conv ops
            # with E4M3 (fp8) A/B, f32 accumulate, f16 C/D. These are ADDED to the
            # pool; conv extract_config only selects them for a conv whose input
            # tensor dtype is actually float8_e4m3, so enabling the flag is inert
            # (never rebinds an f16 conv) until the frontend feeds an E4M3 activation.
            import os as _os_fp8

            if _os_fp8.environ.get("NBT_AIT_FP8_CONV", "0") == "1":
                _generate_sm90_conv3x_fp8_f32acc(cutlass_lib, manifest)
            _drop_sm80_conv_ops(cutlass_lib, manifest)
            # Opt-in validation escape hatch (NOT default): the in-progress gemm
            # 3.x epilogue codegen currently fails to compile some TMA gemm ops
            # (pre-existing, unrelated to conv). Setting AIT_TEMP_DROP_SM90_GEMM3X=1
            # drops CUTLASS-3.x gemm ops so all cutlass gemms use the (correct)
            # SM80 path, allowing an end-to-end bench() validation of the native
            # SM90 conv path. Off by default -> zero effect on gemm behavior.
            import os as _os

            if _os.environ.get("AIT_TEMP_DROP_SM90_GEMM3X", "0") == "1":
                _drop_sm90_3x_gemm_ops(cutlass_lib, manifest)
        elif allow_cutlass_sm90:
            cutlass_lib.generator.GenerateSM90(manifest, args.cuda_version)
            cutlass_lib.generator.GenerateSM80(manifest, args.cuda_version)
            cutlass_lib.extra_operation.GenerateSM80(manifest, args)
        else:
            cutlass_lib.generator.GenerateSM80(manifest, args.cuda_version)
            cutlass_lib.extra_operation.GenerateSM80(manifest, args)
    else:
        try:
            func = getattr(cutlass_lib.generator, "GenerateSM" + arch)
            func(manifest, args.cuda_version)
        except AttributeError as e:
            raise NotImplementedError(
                "Arch " + arch + " is not supported by current cutlass lib."
            ) from e
        try:
            func = getattr(cutlass_lib.extra_operation, "GenerateSM" + arch)
            func(manifest, args)
        except AttributeError:
            _LOGGER.warning("Arch " + arch + " is not supported by extra ops.")

    return _flatten_operations(manifest.operations)


def _generate_sm90_conv3x_f16_f32acc(cutlass_lib, manifest):
    """Generate native SM90 CUTLASS-3.x fprop conv ops with f32 accumulation.

    GenerateSM90_Conv3x only emits f16-output conv kernels that accumulate in
    f16 (its MathInstruction accumulator == data_types['c_type'] == f16). AIT's
    f16 conv reference (and the FORCE=0 SM80 path) accumulates in f32, so f16
    accumulation would break numerical parity on deep-channel trunk convs.

    CreateConvOperator3x takes the accumulator from the tile description's math
    instruction, independently of the C/D element types, so here we build
    MathInstructions with element_accumulator=f32 while keeping C=D=f16. These
    ops carry is_3x=True and are selected by conv extract_config (which requires
    acc==f32). Epilogue fusion (bias / relu / residual) is authored by the conv
    host codegen (backend/cuda/conv2d/common.py).
    """
    gen = cutlass_lib.generator
    lib = cutlass_lib.library

    f16 = lib.DataType.f16
    f32 = lib.DataType.f32
    min_cc, max_cc = 90, 90
    warp_count = [4, 1, 1]
    stages = 0  # auto
    num_mma_per_tile = 4

    schedule_pairs = [
        (
            lib.KernelScheduleType.ImplicitTmaWarpSpecializedSm90,
            lib.EpilogueScheduleType.TmaWarpSpecialized,
        )
    ]
    tile_schedulers = [lib.TileSchedulerType.Default]

    data_types = {
        "a_type": f16,
        "b_type": f16,
        "c_type": f16,
        "d_type": f16,
        "acc_type": f32,
        "epi_type": f32,
        "alignment_A": 8,
        "alignment_B": 8,
        "alignment_C": 8,
    }

    # (mma_m, mma_n, mma_k), cluster_shape. mma_k=16 -> tile_k = 64.
    combos = [
        ((64, 64, 16), (1, 1, 1)),
        ((64, 128, 16), (1, 1, 1)),
        ((128, 64, 16), (1, 1, 1)),
        ((128, 128, 16), (1, 1, 1)),
        ((128, 256, 16), (1, 1, 1)),
        ((256, 128, 16), (1, 1, 1)),
        ((64, 256, 16), (1, 1, 1)),
        ((256, 64, 16), (1, 1, 1)),
    ]

    for mma_shape, cluster_shape in combos:
        math_inst = lib.MathInstruction(
            list(mma_shape),
            f16,
            f16,
            f32,  # element_accumulator = f32
            lib.OpcodeClass.TensorOp,
            lib.MathOperation.multiply_add,
        )
        tile_shape = (mma_shape[0], mma_shape[1], num_mma_per_tile * mma_shape[2])
        tile_description = lib.TileDescription(
            tile_shape, stages, warp_count, math_inst, min_cc, max_cc, cluster_shape
        )
        dims_and_alignments = (((2, 8), (2, 8), (2, 8)),)
        gen.CreateConvOperator3x(
            manifest,
            dims_and_alignments=dims_and_alignments,
            tile_descriptions=[tile_description],
            data_types=data_types,
            schedule_pairs=schedule_pairs,
            tile_schedulers=tile_schedulers,
            conv_kind=lib.ConvKind.Fprop,
        )


def _generate_sm90_conv3x_fp8_f32acc(cutlass_lib, manifest):
    """Generate native SM90 CUTLASS-3.x fprop conv ops with E4M3 (fp8) A/B.

    Mirror of _generate_sm90_conv3x_f16_f32acc but with fp8 e4m3 activation +
    weight, f32 accumulate, f16 C (bias/residual) and D (output feeds the next
    block). Feasibility proven by fp8_conv_spike.cu: the SM90 conv collective
    builder routes e4m3 K-major fprop through the GMMA MMA_*_F32E4M3E4M3_SS_TN
    atom + SM90_TMA_LOAD_IM2COL, so no dispatch/static_assert rejects 8-bit
    element types (see 3rdparty/.../builders/sm90_gmma_builder.inl: it only
    requires the TMA-alignment check, which e4m3 meets at 16-elem alignment).

    fp8 TMA needs 16-byte alignment == 16 e4m3 elements, so A/B alignment is 16
    (vs 8 for f16); C/D stay f16 at alignment 8. The descale (alpha =
    act_scale_inv * w_scale_inv) is applied in the epilogue by the conv host
    codegen (backend/cuda/conv2d/common.py), not here.
    """
    gen = cutlass_lib.generator
    lib = cutlass_lib.library

    e4m3 = lib.DataType.e4m3
    f16 = lib.DataType.f16
    f32 = lib.DataType.f32
    min_cc, max_cc = 90, 90
    warp_count = [4, 1, 1]
    stages = 0  # auto
    num_mma_per_tile = 4

    schedule_pairs = [
        (
            lib.KernelScheduleType.ImplicitTmaWarpSpecializedSm90,
            lib.EpilogueScheduleType.TmaWarpSpecialized,
        )
    ]
    tile_schedulers = [lib.TileSchedulerType.Default]

    # Two output (D) variants:
    #   d_type=f16  : conv output feeds the next block / a residual as f16 (default).
    #   d_type=e4m3 : PRODUCER-EPILOGUE FUSION -- the conv emits E4M3 already scaled for
    #                 the next conv's input, so that next conv skips its fp16->E4M3 cast.
    #                 The fused conv's epilogue folds the next conv's input-scale into
    #                 alpha (= descale * next_act_scale) and its (pre-scaled) bias; the
    #                 ReLu commutes with the positive scale, so it emits the exact E4M3 a
    #                 separate cast would have produced (proven in conv_fp8_fused.cu).
    #   c_type=e4m3 : FULL fusion -- a residual (bias_add) conv whose residual SOURCE is a
    #                 fused producer's E4M3 output; the epilogue dequants it via a runtime
    #                 beta (= out_scale/residual_scale). Bias stays f16 (decoupled from the
    #                 e4m3 source in make_fusion_cpp). (resfuse microbench: correct.)
    base_dt = {
        "a_type": e4m3,
        "b_type": e4m3,
        "acc_type": f32,
        "epi_type": f32,
        "alignment_A": 16,
        "alignment_B": 16,
        "alignment_C": 8,
    }
    data_types_list = [
        {**base_dt, "c_type": f16, "d_type": f16},
        {**base_dt, "c_type": f16, "d_type": e4m3},
        {**base_dt, "c_type": e4m3, "d_type": e4m3},
    ]

    # (mma_m, mma_n, mma_k), cluster_shape. fp8 GMMA K-instruction is 32, but the
    # builder derives the MMA K from the element type; tile_k = num_mma_per_tile *
    # 16 = 64 matches the f16 pool and compiles (verified by the spike at tile_k 64).
    combos = [
        ((64, 64, 16), (1, 1, 1)),
        ((64, 128, 16), (1, 1, 1)),
        ((128, 64, 16), (1, 1, 1)),
        ((128, 128, 16), (1, 1, 1)),
        ((128, 256, 16), (1, 1, 1)),
        ((256, 128, 16), (1, 1, 1)),
        ((64, 256, 16), (1, 1, 1)),
        ((256, 64, 16), (1, 1, 1)),
    ]

    for mma_shape, cluster_shape in combos:
        math_inst = lib.MathInstruction(
            list(mma_shape),
            e4m3,
            e4m3,
            f32,  # element_accumulator = f32
            lib.OpcodeClass.TensorOp,
            lib.MathOperation.multiply_add,
        )
        tile_shape = (mma_shape[0], mma_shape[1], num_mma_per_tile * mma_shape[2])
        tile_description = lib.TileDescription(
            tile_shape, stages, warp_count, math_inst, min_cc, max_cc, cluster_shape
        )
        # A/B alignment 16 (fp8 TMA), C alignment 8 (f16). D alignment is derived by
        # the emitter as 128/sizeof_bits(D) (8 for f16, 16 for e4m3), independent of this.
        dims_and_alignments = (((2, 16), (2, 16), (2, 8)),)
        for data_types in data_types_list:
            gen.CreateConvOperator3x(
                manifest,
                dims_and_alignments=dims_and_alignments,
                tile_descriptions=[tile_description],
                data_types=data_types,
                schedule_pairs=schedule_pairs,
                tile_schedulers=tile_schedulers,
                conv_kind=lib.ConvKind.Fprop,
            )


def _drop_sm90_3x_gemm_ops(cutlass_lib, manifest):
    """Opt-in: drop CUTLASS-3.x (Universal3x) gemm ops, keeping SM80 gemm ops.

    Only used when AIT_TEMP_DROP_SM90_GEMM3X=1 (validation escape hatch). Lets
    the build complete on SM80 gemms while conv uses native SM90 3.x kernels.
    """
    library = cutlass_lib.library
    gemm_kind = getattr(library.OperationKind, "Gemm", None)
    if gemm_kind is None:
        return
    u3x = getattr(library.GemmKind, "Universal3x", None)

    def _keep(ops):
        return [op for op in ops if getattr(op, "gemm_kind", None) != u3x]

    level1 = manifest.operations.get(gemm_kind)
    if not level1:
        return
    values = list(level1.values())
    is_nested = bool(values) and all(isinstance(v, dict) for v in values)
    if is_nested:
        for _min_cc, configs in list(level1.items()):
            for config_name, ops in list(configs.items()):
                kept = _keep(ops)
                if kept:
                    configs[config_name] = kept
                else:
                    del configs[config_name]
    else:
        for config_name, ops in list(level1.items()):
            kept = _keep(ops)
            if kept:
                level1[config_name] = kept
            else:
                del level1[config_name]


def _drop_sm80_conv_ops(cutlass_lib, manifest):
    """Remove SM80 (2.x) convolution ops from the manifest, keeping is_3x ones.

    Under AIT_FORCE_CUTLASS_SM90_KERNELS=1 we want the conv profiler to select
    a native SM90 CUTLASS-3.x conv kernel, not an SM80 fallback. GenerateSM90
    adds is_3x ConvOperation3x ops; GenerateSM80 / extra_operation.GenerateSM80
    add 2.x Conv2dOperation ops into the same OperationKind buckets. Strip the
    non-is_3x conv ops so only the 3.x pool remains. GEMM ops are left intact.

    Handles both the cutlass >= 3.2 nested layout
    (operations[kind][min_cc][config] -> [ops]) and the historical flat layout
    (operations[kind][config] -> [ops]).
    """
    library = cutlass_lib.library
    conv_kinds = set()
    for name in ("Conv2d", "Conv3d"):
        kind = getattr(library.OperationKind, name, None)
        if kind is not None:
            conv_kinds.add(kind)

    def _keep(ops):
        return [op for op in ops if getattr(op, "is_3x", False)]

    for kind in conv_kinds:
        level1 = manifest.operations.get(kind)
        if not level1:
            continue
        values = list(level1.values())
        is_nested = bool(values) and all(isinstance(v, dict) for v in values)
        if is_nested:
            for _min_cc, configs in list(level1.items()):
                for config_name, ops in list(configs.items()):
                    kept = _keep(ops)
                    if kept:
                        configs[config_name] = kept
                    else:
                        del configs[config_name]
        else:
            for config_name, ops in list(level1.items()):
                kept = _keep(ops)
                if kept:
                    level1[config_name] = kept
                else:
                    del level1[config_name]


def _flatten_operations(operations):
    """Normalize the cutlass manifest operations layout.

    cutlass >= 3.2 nests operations by minimum compute capability:
        operations[kind][min_cc][configuration_name] -> [Operation, ...]
    while the AITemplate op extractors expect the historical layout:
        operations[kind][configuration_name] -> [Operation, ...]

    Collapse the min_cc level (merging across compute capabilities) so the
    downstream extractors keep working regardless of the cutlass version.
    """
    flattened = {}
    for kind, level1 in operations.items():
        values = list(level1.values())
        if values and all(isinstance(v, dict) for v in values):
            merged = OrderedDict()
            for _min_cc, configs in level1.items():
                for config_name, ops in configs.items():
                    merged.setdefault(config_name, []).extend(ops)
            flattened[kind] = merged
        else:
            flattened[kind] = level1
    return flattened
