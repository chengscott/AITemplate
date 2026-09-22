# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import math
import tempfile
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target


class SM90ShortAttentionTestCase(unittest.TestCase):
    @torch.no_grad()
    def test_short_attention_and_fallbacks(self):
        target = detect_target(use_cutedsl_attention=True, use_fp16_acc=False)
        if target.name() != "cuda" or target._arch != "90":
            self.skipTest("SM90-specific attention selection")
        cases = [(1, 16, False), (63, 16, False), (81, 16, False),
                 (128, 16, False), (129, 16, False), (81, 16, True),
                 (81, 32, False)]
        batch = IntVar([1, 17], name="batch")
        outputs = []
        for i, (seq, dim, causal) in enumerate(cases):
            x = Tensor([batch, seq, 3, 3, dim], name=f"x{i}", is_input=True)
            y = ops.flash_attention(17, 0, seq, causal)(x)
            y._attrs.update(name=f"y{i}", is_output=True)
            outputs.append(y)
        with tempfile.TemporaryDirectory(prefix="ait_sm90_short_attention_") as directory:
            model = compile_model(outputs, target, directory, "model")
            torch.manual_seed(99)
            for n in (1, 3, 17):
                inputs, expected = {}, []
                for i, (seq, dim, causal) in enumerate(cases):
                    x = torch.randn(n, seq, 3, 3, dim, device="cuda", dtype=torch.float16)
                    inputs[f"x{i}"] = x
                    q, k, v = [t.float().transpose(1, 2) for t in x.unbind(2)]
                    scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
                    if causal:
                        mask = torch.ones(seq, seq, device="cuda", dtype=torch.bool).triu(1)
                        scores.masked_fill_(mask, float("-inf"))
                    expected.append((scores.softmax(-1) @ v).transpose(1, 2).contiguous().half())
                actual = [torch.empty_like(y) for y in expected]
                model.run_with_tensors(inputs, actual)
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, atol=0.002, rtol=0.002)


if __name__ == "__main__":
    unittest.main()
