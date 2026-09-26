"""Compiler profiling hooks shared by the native projection fusions."""


class NativeFusionProfiler:
    def gen_profiler(self, workdir="./", dynamic_profiling_strategy=None):
        from aitemplate.backend.cuda.gemm_universal.native_fused_gemm_profiler import (
            gen_profiler,
        )

        return gen_profiler(self._attrs, workdir)

    def profile(self, workdir="./", devices=None, **kwargs):
        from aitemplate.backend.cuda.gemm_universal.native_fused_gemm_profiler import (
            profile,
        )

        profile(self._attrs, devices)
