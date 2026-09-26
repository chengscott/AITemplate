"""CuTe AOT block-scaled SwiGLU producer and residual consumer."""

from pathlib import Path
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from quack.activation import swiglu
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import unpack, F2
from quack.epilogue.ops import ColVecLoad, TileStore
from quack.epilogue.quantize_out import BlockScaleFactorStore
from quack.compile_utils import make_fake_tensor
from quack.gemm_tvm_ffi_utils import make_fake_sf_tensor
from quack.tile_scheduler import TileSchedulerOptions


@gemm_epilogue(
    outputs=(
        TileStore(
            "postact",
            gated=True,
            quant=BlockScaleFactorStore("postact_sf", output="postact"),
        ),
    ),
    ops={"rstd": ColVecLoad("rstd")},
    mode="acc_pair",
)
def quant_swiglu(acc, rstd):
    # Preserve the existing FP16 gate/up materialization's rounding.
    gate, up = unpack(acc)
    if cutlass.const_expr(isinstance(gate, F2)):
        gate = F2(
            gate.lo.to(cutlass.Float16).to(cutlass.Float32),
            gate.hi.to(cutlass.Float16).to(cutlass.Float32),
        )
        up = F2(
            up.lo.to(cutlass.Float16).to(cutlass.Float32),
            up.hi.to(cutlass.Float16).to(cutlass.Float32),
        )
    else:
        gate = gate.to(cutlass.Float16).to(cutlass.Float32)
        up = up.to(cutlass.Float16).to(cutlass.Float32)
    value = swiglu(gate * rstd, up * rstd)
    if cutlass.const_expr(isinstance(value, F2)):
        value = F2(
            value.lo.to(cutlass.Float16).to(cutlass.Float32),
            value.hi.to(cutlass.Float16).to(cutlass.Float32),
        )
    else:
        value = value.to(cutlass.Float16).to(cutlass.Float32)
    return {"postact": value}


@gemm_epilogue()
def residual(acc, c):
    return {"D": acc + c}


class _FfnAot:
    def __init__(self, kernel, epi, clusters):
        self.kernel, self.epi, self.clusters = kernel, epi, clusters

    @cute.jit
    def __call__(self, A, B, R, H, SA, SB, SH, stream: cuda.CUstream):
        epi = self.epi(rstd=R, postact=H, postact_sf=SH)
        scheduler = TileSchedulerOptions(
            max_active_clusters=cutlass.Int32(self.clusters),
            max_swizzle_size=cutlass.Int32(8),
        )
        self.kernel(A, B, None, None, epi, scheduler, None, stream, SA, SB)


class _DownAot:
    def __init__(self, kernel, epi, clusters):
        self.kernel, self.epi, self.clusters = kernel, epi, clusters

    @cute.jit
    def __call__(self, A, B, C, D, SA, SB, stream: cuda.CUstream):
        scheduler = TileSchedulerOptions(
            max_active_clusters=cutlass.Int32(self.clusters),
            max_swizzle_size=cutlass.Int32(8),
        )
        self.kernel(A, B, D, C, self.epi(), scheduler, None, stream, SA, SB)


def export(directory, name, producer, tile_n):
    k, n = (192, 1152) if producer else (576, 192)
    x = torch.empty(81, k, device="cuda", dtype=torch.float8_e4m3fn)
    w = torch.empty(n, k, device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.empty(
        1, 1, (k + 127) // 128, 32, 4, 4, device="cuda", dtype=torch.float8_e8m0fnu
    )
    sb = torch.empty(
        1,
        (n + 127) // 128,
        (k + 127) // 128,
        32,
        4,
        4,
        device="cuda",
        dtype=torch.float8_e8m0fnu,
    )
    r = torch.empty(81, device="cuda", dtype=torch.float16)
    h = torch.empty(81, 576, device="cuda", dtype=torch.float8_e4m3fn)
    sh = torch.empty(1, 1, 5, 32, 4, 4, device="cuda", dtype=torch.float8_e8m0fnu)
    y = torch.empty(81, n, device="cuda", dtype=torch.float16)
    epi = quant_swiglu if producer else residual
    plan = epi.gemm(
        A=x,
        B=w,
        C=None if producer else y,
        D=None if producer else y,
        SFA=sa,
        SFB=sb,
        bs_format_a="mxfp8",
        bs_format_b="mxfp8",
        epi_args={"rstd": r, "postact": h, "postact_sf": sh} if producer else {},
        tile_M=128,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        is_dynamic_persistent=False,
        _launch=False,
    )
    kernel = plan.gemm_cls(
        cutlass.Float32,
        cutlass.Float8E4M3FN,
        (128, tile_n),
        (1, 1, 1),
        sf_vec_size=32,
        gather_A=False,
        use_clc_persistence=False,
    )
    kernel.b_transposed = kernel.a_transposed = kernel.cd_transposed = False
    kernel.cd_packed = None
    m, nn, kk = cute.sym_int(), cute.sym_int(), cute.sym_int()
    A = make_fake_tensor(cutlass.Float8E4M3FN, (m, kk), divisibility=16)
    B = make_fake_tensor(cutlass.Float8E4M3FN, (nn, kk), divisibility=16)
    SA = make_fake_sf_tensor(cutlass.Float8E8M0FNU, 1)
    SB = make_fake_sf_tensor(cutlass.Float8E8M0FNU, 1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if producer:
        R = make_fake_tensor(cutlass.Float16, (m,), divisibility=2)
        H = make_fake_tensor(cutlass.Float8E4M3FN, (m, cute.sym_int()), divisibility=16)
        SH = make_fake_sf_tensor(cutlass.Float8E8M0FNU, 1)
        compiled = cute.compile(
            _FfnAot(kernel, plan.gemm_cls.EpilogueArguments, plan.max_active_clusters),
            A,
            B,
            R,
            H,
            SA,
            SB,
            SH,
            stream,
        )
    else:
        C = make_fake_tensor(cutlass.Float16, (m, nn), divisibility=8)
        D = make_fake_tensor(cutlass.Float16, (m, nn), divisibility=8)
        compiled = cute.compile(
            _DownAot(kernel, plan.gemm_cls.EpilogueArguments, plan.max_active_clusters),
            A,
            B,
            C,
            D,
            SA,
            SB,
            stream,
        )
    Path(directory).mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(file_path=str(directory), file_name=name)
    return str(Path(directory) / (name + ".o"))
