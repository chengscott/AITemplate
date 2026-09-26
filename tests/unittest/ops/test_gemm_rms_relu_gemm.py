"""Accuracy and dynamic CUDA-graph coverage for RMS/ReLU projection fusion."""

import tempfile
import unittest
import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target


class GemmRmsReluGemmTest(unittest.TestCase):
    @torch.no_grad()
    def test_boundary(self):
        target = detect_target(use_fp16_acc=False)
        if target.name() != "cuda" or target._arch not in ("90", "100"):
            self.skipTest("Requires SM90 or SM100")
        torch.backends.cuda.matmul.allow_tf32 = False
        batch = IntVar([1, 2048], name="batch")
        shapes = {
            "x": [batch, 81, 576],
            "w": [192, 576],
            "skip": [batch, 81, 192],
            "gamma": [192],
            "up": [384, 192],
            "outer": [batch, 81, 384],
        }
        inputs = {
            name: Tensor(shape, name=name, is_input=True)
            for name, shape in shapes.items()
        }
        eps = 3e-4
        out = ops.gemm_rms_relu_gemm(eps=eps)(*inputs.values())
        out._attrs.update(name="out", is_output=True)
        with tempfile.TemporaryDirectory(prefix="ait_rms_boundary_") as work:
            model = compile_model(out, target, work, "model")
            buffers = {}
            for b in (1, 4, 8, 16, 17, 22, 23, 31, 32, 33, 128, 129, 512, 2048, 1):
                torch.manual_seed(20260927 + b)
                values = {
                    name: torch.randn(
                        *[b if d is batch else d for d in shape],
                        device="cuda",
                        dtype=torch.float16
                    )
                    for name, shape in shapes.items()
                }
                for name in ("w", "up"):
                    values[name] *= 0.05
                for name in ("x", "skip", "outer"):
                    values[name] *= 0.25
                values["x"][:, 0].zero_()
                values["skip"][:, 0].zero_()
                z = (
                    (
                        values["x"].float() @ values["w"].float().T
                        + values["skip"].float()
                    )
                    .half()
                    .float()
                )
                normalized = (
                    (
                        z
                        * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
                        * values["gamma"].float()
                    )
                    .relu()
                    .half()
                )
                expected = (
                    normalized.float() @ values["up"].float().T
                    + values["outer"].float()
                ).half()
                actual = torch.empty_like(expected)
                # Graph replay requires stable input and output addresses per shape.
                if b in buffers:
                    saved, actual = buffers[b]
                    for name in values:
                        saved[name].copy_(values[name])
                    values = saved
                else:
                    buffers[b] = (values, actual)
                model.run_with_tensors(values, [actual], graph_mode=True)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)


if __name__ == "__main__":
    unittest.main()
