# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import tempfile
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target


class SM100StaticElementwiseTestCase(unittest.TestCase):
    @torch.no_grad()
    def test_static_widths_and_partial_warps(self):
        target = detect_target(use_fp16_acc=False)
        if target.name() != "cuda" or target._arch not in ("80", "90", "100"):
            self.skipTest("SM80/SM90/SM100 specialization")
        cases = [(8, 7), (192, 576), (384, 256), (1024, 1024)]
        rows = IntVar([1, 257], name="rows")
        outputs = []
        for i, (channels, ffn) in enumerate(cases):
            x = Tensor([rows, channels], name=f"x{i}", is_input=True)
            gamma = Tensor([channels], name=f"g{i}", is_input=True)
            packed = Tensor([rows, 2 * ffn], name=f"p{i}", is_input=True)
            scale = Tensor([rows, 1], name=f"s{i}", is_input=True)
            outputs.extend([
                ops.rmsnorm()(x, gamma), ops.rmsnorm(relu=True)(x, gamma),
                ops.rms_reduce()(x), ops.swiglu()(packed), ops.swiglu()(packed, scale),
            ])
        for i, output in enumerate(outputs):
            output._attrs.update(name=f"out{i}", is_output=True)
        with tempfile.TemporaryDirectory(prefix="ait_static_widths_") as directory:
            model = compile_model(outputs, target, directory, "model")
            torch.manual_seed(18)
            for n in (1, 7, 257):
                inputs, expected = {}, []
                for i, (channels, ffn) in enumerate(cases):
                    x = torch.randn(n, channels, device="cuda", dtype=torch.float16)
                    x[0].zero_()  # eps handling, including a whole zero row
                    gamma = torch.randn(channels, device="cuda", dtype=torch.float16)
                    packed = torch.randn(n, 2 * ffn, device="cuda", dtype=torch.float16)
                    scale = torch.rand(n, 1, device="cuda", dtype=torch.float16) + 0.5
                    inputs.update({f"x{i}": x, f"g{i}": gamma, f"p{i}": packed, f"s{i}": scale})
                    rrms = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                    norm = x.float() * rrms * gamma.float()
                    gate, up = packed.float().chunk(2, dim=-1)
                    expected.extend([
                        norm.half(), norm.relu().half(), rrms.half(),
                        (torch.nn.functional.silu(gate) * up).half(),
                        (torch.nn.functional.silu(gate * scale.float()) * (up * scale.float())).half(),
                    ])
                actual = [torch.empty_like(value) for value in expected]
                model.run_with_tensors(inputs, actual)
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, atol=0.002, rtol=0.002)


if __name__ == "__main__":
    unittest.main()
