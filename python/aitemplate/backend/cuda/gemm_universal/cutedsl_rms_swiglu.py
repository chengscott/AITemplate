"""CuTe/Quack AOT epilogues for residual, RMS scaling, and SwiGLU."""

from pathlib import Path
import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from quack.activation import swiglu
from quack.compile_utils import make_fake_tensor
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.ops import ColVecLoad, ColVecReduce
from quack.epilogue.math import unpack, F2
from quack.tile_scheduler import TileSchedulerOptions


@gemm_epilogue(outputs=("postact",), ops={"rstd": ColVecLoad("rstd")}, mode="acc_pair")
def _ffn_epilogue(acc, rstd):
    gate, up = unpack(acc * rstd)
    return {"postact": swiglu(gate, up)}


@gemm_epilogue(reduces={"sqsum": ColVecReduce("sqsum")}, vectorize=False)
def _residual_epilogue(acc, c):
    value = acc + c
    if cutlass.const_expr(isinstance(value, F2)):
        value = F2(
            value.lo.to(cutlass.Float16).to(cutlass.Float32),
            value.hi.to(cutlass.Float16).to(cutlass.Float32),
        )
    else:
        value = value.to(cutlass.Float16).to(cutlass.Float32)
    return {"D": value, "sqsum": value * value}


class _RmsReduce(ColVecReduce):
    def __init__(self, name, eps):
        super().__init__(name)
        self.eps = eps

    def config_key(self):
        return (*super().config_key(), self.eps)

    @cute.jit
    def _finalize(self, vals):
        inv = cute.math.rsqrt(vals[0] / 192.0 + self.eps)
        return inv.to(cutlass.Float16).to(cutlass.Float32)


def _make_residual_rms(eps):
    # This reduction is valid only when a CTA covers the entire output width.
    @gemm_epilogue(reduces={"sqsum": _RmsReduce("sqsum", eps)}, vectorize=False)
    def epilogue(acc, c):
        value = acc + c
        if cutlass.const_expr(isinstance(value, F2)):
            value = F2(
                value.lo.to(cutlass.Float16).to(cutlass.Float32),
                value.hi.to(cutlass.Float16).to(cutlass.Float32),
            )
        else:
            value = value.to(cutlass.Float16).to(cutlass.Float32)
        return {"D": value, "sqsum": value * value}

    return epilogue


class _FfnAot:
    def __init__(self, kernel, epi_cls, clusters):
        self.kernel, self.epi_cls, self.clusters = kernel, epi_cls, clusters

    @cute.jit
    def __call__(self, A, B, R, O, stream: cuda.CUstream):
        epi = self.epi_cls(rstd=R, postact=O)
        scheduler = TileSchedulerOptions(
            max_active_clusters=cutlass.Int32(self.clusters),
            max_swizzle_size=cutlass.Int32(8),
        )
        self.kernel(A, B, None, None, epi, scheduler, None, stream, None, None)


class _ResidualAot:
    def __init__(self, kernel, epi_cls, clusters):
        self.kernel, self.epi_cls, self.clusters = kernel, epi_cls, clusters

    @cute.jit
    def __call__(self, A, B, C, D, P, stream: cuda.CUstream):
        epi = self.epi_cls(sqsum=P)
        scheduler = TileSchedulerOptions(
            max_active_clusters=cutlass.Int32(self.clusters),
            max_swizzle_size=cutlass.Int32(8),
        )
        self.kernel(A, B, D, C, epi, scheduler, None, stream, None, None)


def export(
    directory,
    name,
    arch,
    tile_n,
    residual=False,
    tile_m=128,
    pingpong=None,
    direct_rms=False,
    eps=1e-6,
):
    """Export a dynamic-row C interface; Python/Quack are build dependencies only."""
    if direct_rms and (not residual or tile_n < 192):
        raise ValueError("Direct RMS requires a full-width residual tile")
    if pingpong is None:
        pingpong = tile_n != 64
    x = torch.empty(81, 192, device="cuda", dtype=torch.float16)
    w = torch.empty(192 if residual else 1152, 192, device="cuda", dtype=torch.float16)
    out = torch.empty(81, 192 if residual else 576, device="cuda", dtype=torch.float16)
    aux = torch.empty(
        (81, (192 + tile_n - 1) // tile_n) if residual else (81,),
        device="cuda",
        dtype=torch.float32,
    )
    epi = (
        (_make_residual_rms(eps) if direct_rms else _residual_epilogue)
        if residual
        else _ffn_epilogue
    )
    kwargs = dict(
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        pingpong=pingpong,
        is_dynamic_persistent=False,
    )
    # Resolve the epilogue schema through the public plan API. Reconstruct its
    # kernel with a flat C-exportable signature instead of TVM-FFI structures.
    plan = epi.gemm(
        A=x,
        B=w,
        C=x if residual else None,
        D=out if residual else None,
        epi_args={"sqsum": aux} if residual else {"rstd": aux, "postact": out},
        concat_layout=None if residual else ("B",),
        _launch=False,
        **kwargs
    )
    kernel_kwargs = dict(gather_A=False, concat_layout=None if residual else ("B",))
    if arch == 90:
        kernel_kwargs.update(pingpong=pingpong, is_persistent=True)
    else:
        kernel_kwargs.update(use_clc_persistence=False)
    kernel = plan.gemm_cls(
        cutlass.Float32, cutlass.Float16, (tile_m, tile_n), (1, 1, 1), **kernel_kwargs
    )
    kernel.b_transposed = kernel.a_transposed = kernel.cd_transposed = False
    kernel.cd_packed = None
    fake = lambda dtype, dims, align: make_fake_tensor(dtype, dims, divisibility=align)
    m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int()
    A = fake(cutlass.Float16, (m, k), 8)
    B = fake(cutlass.Float16, (n, k), 8)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if residual:
        C = fake(cutlass.Float16, (m, n), 8)
        D = fake(cutlass.Float16, (m, n), 8)
        P = fake(cutlass.Float32, (m, cute.sym_int()), 1)
        wrapper = _ResidualAot(
            kernel, plan.gemm_cls.EpilogueArguments, plan.max_active_clusters
        )
        compiled = cute.compile(wrapper, A, B, C, D, P, stream)
    else:
        R = fake(cutlass.Float32, (m,), 1)
        O = fake(cutlass.Float16, (m, cute.sym_int(divisibility=8)), 8)
        wrapper = _FfnAot(
            kernel, plan.gemm_cls.EpilogueArguments, plan.max_active_clusters
        )
        compiled = cute.compile(wrapper, A, B, R, O, stream)
    Path(directory).mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(file_path=str(directory), file_name=name)
    return str(Path(directory) / (name + ".o"))
