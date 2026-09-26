"""Native CUTLASS implementations of specialized projection fusions.

Automatic dispatch uses native CUTLASS on SM80/SM90/SM100.
AIT_FUSED_GEMM_BACKEND explicitly selects either implementation.
"""

import os


def target_arch():
    from aitemplate.backend.target import Target

    try:
        return int(Target.current()._arch)
    except RuntimeError:
        return None


def enabled():
    choice = os.environ.get("AIT_FUSED_GEMM_BACKEND", "auto")
    if choice not in ("auto", "cutlass", "quack"):
        raise ValueError("AIT_FUSED_GEMM_BACKEND must be auto, cutlass, or quack")
    return choice == "cutlass" or (choice == "auto" and target_arch() in (80, 90, 100))


def boundary(attrs, arch):
    from .gemm_rms_relu_gemm import _signature
    from .native_fused_gemm_profiler import function

    return function(attrs, arch, _signature(attrs["name"]))


def mxfp8(attrs):
    from .gemm_mxfp8_swiglu import _signature
    from .native_fused_gemm_profiler import function

    return function(attrs, 100, _signature(attrs["name"]))


def swiglu(attrs, arch):
    from .gemm_rms_swiglu import _signature
    from .native_fused_gemm_profiler import function

    return function(attrs, arch, _signature(attrs["name"]))
