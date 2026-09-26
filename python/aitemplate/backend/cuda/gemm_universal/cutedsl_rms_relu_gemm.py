"""CuTe AOT kernels for a residual/RMS/ReLU projection boundary."""

from pathlib import Path
import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from quack.activation import relu
from quack.compile_utils import make_fake_tensor
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad
from quack.epilogue.math import F2
from quack.tile_scheduler import TileSchedulerOptions


class _Rms(ColVecReduce):
    def __init__(self, name, eps):
        super().__init__(name)
        self.eps = eps

    def config_key(self):
        return (*super().config_key(), self.eps)

    @cute.jit
    def _finalize(self, vals):
        return cute.math.rsqrt(vals[0] / 192.0 + self.eps)


def _producer(eps):
    @gemm_epilogue(
        ops={"weight": RowVecLoad("weight")},
        reduces={"rstd": _Rms("rstd", eps)},
        vectorize=False,
    )
    def epilogue(acc, c, weight):
        value = acc + c
        if cutlass.const_expr(isinstance(value, F2)):
            value = F2(
                value.lo.to(cutlass.Float16).to(cutlass.Float32),
                value.hi.to(cutlass.Float16).to(cutlass.Float32),
            )
        else:
            value = value.to(cutlass.Float16).to(cutlass.Float32)
        return {"D": relu(value * weight), "rstd": value * value}

    return epilogue


@gemm_epilogue(ops={"rstd": ColVecLoad("rstd")})
def _consumer(acc, c, rstd):
    return {"D": acc * rstd + c}


class _Aot:
    def __init__(self, kernel, epi_cls, clusters, producer):
        self.kernel, self.epi_cls = kernel, epi_cls
        self.clusters, self.producer = clusters, producer

    @cute.jit
    def __call__(self, A, B, C, D, R, G, stream: cuda.CUstream):
        if cutlass.const_expr(self.producer):
            epi = self.epi_cls(rstd=R, weight=G)
        else:
            epi = self.epi_cls(rstd=R)
        scheduler = TileSchedulerOptions(
            max_active_clusters=cutlass.Int32(self.clusters),
            max_swizzle_size=cutlass.Int32(8),
        )
        self.kernel(A, B, D, C, epi, scheduler, None, stream, None, None)


def export(directory, name, arch, producer, tile_m, tile_n, pingpong, eps):
    if producer and tile_n != 192:
        raise ValueError("RMS reduction requires a full-width tile")
    k, n = (576, 192) if producer else (192, 384)
    x = torch.empty(81, k, device="cuda", dtype=torch.float16)
    w = torch.empty(n, k, device="cuda", dtype=torch.float16)
    y = torch.empty(81, n, device="cuda", dtype=torch.float16)
    r = torch.empty((81, 1) if producer else (81,), device="cuda", dtype=torch.float32)
    g = torch.empty(192, device="cuda", dtype=torch.float16)
    epi = _producer(eps) if producer else _consumer
    plan = epi.gemm(
        A=x,
        B=w,
        C=y,
        D=y,
        epi_args={"rstd": r, "weight": g} if producer else {"rstd": r},
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        pingpong=pingpong,
        is_dynamic_persistent=False,
        _launch=False,
    )
    kwargs = dict(gather_A=False)
    if arch == 90:
        kwargs.update(pingpong=pingpong, is_persistent=True)
    else:
        kwargs.update(use_clc_persistence=False)
    kernel = plan.gemm_cls(
        cutlass.Float32, cutlass.Float16, (tile_m, tile_n), (1, 1, 1), **kwargs
    )
    kernel.b_transposed = kernel.a_transposed = kernel.cd_transposed = False
    kernel.cd_packed = None
    fake = lambda dtype, shape, align: make_fake_tensor(
        dtype, shape, divisibility=align
    )
    m, nn, kk = cute.sym_int(), cute.sym_int(), cute.sym_int()
    A = fake(cutlass.Float16, (m, kk), 8)
    B = fake(cutlass.Float16, (nn, kk), 8)
    C = fake(cutlass.Float16, (m, nn), 8)
    D = fake(cutlass.Float16, (m, nn), 8)
    R = fake(cutlass.Float32, (m, cute.sym_int()) if producer else (m,), 1)
    G = fake(cutlass.Float16, (cute.sym_int(),), 8)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        _Aot(
            kernel, plan.gemm_cls.EpilogueArguments, plan.max_active_clusters, producer
        ),
        A,
        B,
        C,
        D,
        R,
        G,
        stream,
    )
    Path(directory).mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(file_path=str(directory), file_name=name)
    return str(Path(directory) / (name + ".o"))
