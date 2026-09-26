"""MXFP8 producer fusion, scale packing, and constant-refolding coverage."""

import tempfile
import unittest
import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target


def quantize(x):
    m, k = x.shape
    blocks = x.float().reshape(m, k // 32, 32)
    amax = blocks.abs().amax(-1)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    q = (blocks / scale[..., None]).reshape(m, k).to(torch.float8_e4m3fn)
    dq = (q.float().reshape(m, k // 32, 32) * scale[..., None]).reshape(m, k)
    packed = torch.zeros(
        (m + 127) // 128, (k + 127) // 128, 32, 4, 4, device=x.device, dtype=torch.uint8
    )
    rows = torch.arange(m, device=x.device)[:, None]
    cols = torch.arange(k // 32, device=x.device)[None, :]
    packed[rows // 128, cols // 4, rows % 32, (rows % 128) // 32, cols % 4] = (
        torch.log2(scale) + 127
    ).to(torch.uint8)
    return q, packed.flatten().view(torch.float8_e4m3fn), dq


class GemmMxfp8SwigluTest(unittest.TestCase):
    @torch.no_grad()
    def test_fusion(self):
        target = detect_target(use_fp16_acc=False)
        if target.name() != "cuda" or target._arch != "100":
            self.skipTest("Requires SM100")
        torch.backends.cuda.matmul.allow_tf32 = False
        b = IntVar([1, 2048], name="batch")
        x = Tensor([b, 81, 192], name="x", is_input=True)
        shapes = {"w": [1152, 192], "sf": [9216], "down": [192, 576], "down_sf": [5120]}
        constants = {
            n: Tensor(s, dtype="float8_e4m3", name=n) for n, s in shapes.items()
        }
        out = ops.gemm_mxfp8_swiglu()(x, *constants.values())
        out._attrs.update(name="out", is_output=True)
        ref_w = Tensor([1152, 192], dtype="float8_e4m3", name="ref_w")
        ref_sf = Tensor([9216], dtype="float8_e4m3", name="ref_sf")
        raw, inv = ops.gemm_rcr_mxfp8()(x, ref_w, ref_sf, norm="rms_out")
        baseline = ops.gemm_rcr_mxfp8()(
            raw, constants["down"], constants["down_sf"], x, norm="swiglu", rrms=inv
        )
        baseline._attrs.update(name="baseline", is_output=True)
        with tempfile.TemporaryDirectory(prefix="ait_mxfp8_ffn_") as work:
            model = compile_model([out, baseline], target, work, "model")
            buffers = {}
            stable_weights = None
            for seed in (20260927, 20260928):
                torch.manual_seed(seed)
                w, sf, dw = quantize(
                    torch.randn(1152, 192, device="cuda", dtype=torch.float16) * 0.05
                )
                down, down_sf, dd = quantize(
                    torch.randn(192, 576, device="cuda", dtype=torch.float16) * 0.05
                )
                values = dict(
                    w=w, sf=sf, down=down, down_sf=down_sf, ref_w=w, ref_sf=sf
                )
                if stable_weights is not None:
                    for name, tensor in values.items():
                        stable_weights[name].copy_(tensor)
                    values = stable_weights
                else:
                    stable_weights = values
                model.set_many_constants_with_tensors(values)
                model.fold_constants(sync=True)
                for batch in (
                    (1, 8, 9, 128, 129, 512, 2048, 1) if seed == 20260927 else (1, 128)
                ):
                    source = (
                        torch.randn(batch, 81, 192, device="cuda", dtype=torch.float16)
                        * 0.25
                    )
                    source[:, 0].zero_()
                    flat = source.reshape(-1, 192)
                    _, _, dx = quantize(flat)
                    inv = (
                        torch.rsqrt(flat.float().square().mean(-1, keepdim=True) + 1e-6)
                        .half()
                        .float()
                    )
                    gate, up = ((dx @ dw.T).half().float() * inv).chunk(2, -1)
                    hidden = (torch.nn.functional.silu(gate) * up).half()
                    _, _, dh = quantize(hidden)
                    expected = (dh @ dd.T + flat.float()).half().reshape_as(source)
                    actual = torch.empty_like(expected)
                    original = torch.empty_like(expected)
                    # The runtime caches CUDA graphs by shape; retain their addresses.
                    if batch in buffers:
                        saved, actual, original = buffers[batch]
                        saved.copy_(source)
                        source = saved
                    else:
                        buffers[batch] = (source, actual, original)
                    if batch == 1:
                        model.run_with_tensors(
                            [source], [actual, original], graph_mode=False
                        )
                    model.run_with_tensors(
                        [source], [actual, original], graph_mode=True
                    )
                    torch.testing.assert_close(actual, original, atol=0.01, rtol=0.01)
                    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
