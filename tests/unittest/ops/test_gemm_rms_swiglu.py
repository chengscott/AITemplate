"""Numerical and CUDA-graph transition coverage for RMS GEMM fusions."""

import tempfile
import os
import unittest
import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target


class GemmRmsSwigluTest(unittest.TestCase):
    def test_fusions(self):
        self._check_fusions(False)

    def test_attention_composition(self):
        self._check_fusions(True)

    @torch.no_grad()
    def _check_fusions(self, include_attention):
        target = detect_target(use_fp16_acc=False)
        if target.name() != "cuda" or target._arch not in ("80", "90", "100"):
            self.skipTest("Requires SM80, SM90 or SM100")
        if target._arch == "80":
            if os.environ.get("AIT_FUSED_GEMM_BACKEND", "auto") == "quack":
                self.skipTest("SM80 fusion requires the native backend")
            if include_attention:
                self.skipTest("This composition test uses SM90/SM100 attention")
        torch.backends.cuda.matmul.allow_tf32 = False
        b = IntVar([1, 2048], name="batch")
        x = Tensor([b, 81, 192], name="x", is_input=True)
        w = Tensor([1152, 192], name="w", is_input=True)
        r = Tensor([b, 81, 192], name="r", is_input=True)
        p = Tensor([192, 192], name="p", is_input=True)
        plain = ops.gemm_rms_swiglu()(x, w)
        joint_eps = 3e-4
        updated, joint = ops.gemm_rms_swiglu(eps=joint_eps)(x, w, r, p)
        qkv = [
            Tensor([b, 81, 12, 16], name=name, is_input=True)
            for name in ("q", "k", "v")
        ]
        outputs = [plain, updated, joint]
        names = ["plain", "updated", "joint"]
        if include_attention:
            outputs.append(ops.flash_attention_qkv(seq_len=81)(*qkv))
            names.append("attention")
        for name, t in zip(names, outputs):
            t._attrs.update(name=name, is_output=True)
        with tempfile.TemporaryDirectory(prefix="ait_fused_ffn_") as work:
            model = compile_model(outputs, target, work, "model")
            buffers = {}
            for batch in (
                1,
                4,
                5,
                8,
                9,
                16,
                17,
                22,
                23,
                32,
                33,
                64,
                65,
                128,
                129,
                256,
                767,
                768,
                769,
                1024,
                2048,
                1,
            ):
                torch.manual_seed(20260924 + batch)
                values = dict(
                    x=torch.randn(batch, 81, 192, device="cuda", dtype=torch.float16)
                    * 0.25,
                    w=torch.randn(1152, 192, device="cuda", dtype=torch.float16) * 0.05,
                    r=torch.randn(batch, 81, 192, device="cuda", dtype=torch.float16)
                    * 0.25,
                    p=torch.randn(192, 192, device="cuda", dtype=torch.float16) * 0.05,
                )
                if include_attention:
                    values.update(
                        {
                            name: torch.randn(
                                batch, 81, 12, 16, device="cuda", dtype=torch.float16
                            )
                            * 0.25
                            for name in ("q", "k", "v")
                        }
                    )
                values["x"][:, 0].zero_()
                values["r"][:, 0].zero_()
                expected_updated = (
                    values["x"].float() @ values["p"].float().T + values["r"].float()
                ).half()

                def reference(source, eps=1e-6):
                    inv = (
                        torch.rsqrt(
                            source.float().square().mean(-1, keepdim=True) + eps
                        )
                        .half()
                        .float()
                    )
                    gate, up = ((source.float() @ values["w"].float().T) * inv).chunk(
                        2, -1
                    )
                    return (torch.nn.functional.silu(gate) * up).half()

                expected = [
                    reference(values["x"]),
                    expected_updated,
                    reference(expected_updated, joint_eps),
                ]
                if include_attention:
                    expected.append(
                        torch.nn.functional.scaled_dot_product_attention(
                            *[
                                values[name].float().transpose(1, 2)
                                for name in ("q", "k", "v")
                            ]
                        )
                        .transpose(1, 2)
                        .half()
                    )
                actual = [torch.empty_like(t) for t in expected]
                # Replayed graphs retain tensor addresses for each shape.
                if batch in buffers:
                    saved, actual = buffers[batch]
                    for name in values:
                        saved[name].copy_(values[name])
                    values = saved
                else:
                    buffers[batch] = (values, actual)
                if batch == 1:
                    model.run_with_tensors(values, actual, graph_mode=False)
                model.run_with_tensors(values, actual, graph_mode=True)
                for out, ref in zip(actual, expected):
                    torch.testing.assert_close(out, ref, atol=0.002, rtol=0.002)


if __name__ == "__main__":
    unittest.main()
