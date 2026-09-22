# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import os
import unittest
from unittest import mock

from aitemplate.backend import registry
from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target


class SM90ChannelTilesTestCase(unittest.TestCase):
    def test_convolution_candidate_scope_and_emission(self):
        with mock.patch.dict(os.environ, {"AIT_FORCE_CUTLASS_SM90_KERNELS": "1"}):
            target = detect_target(use_fp16_acc=False)
            if target.name() != "cuda" or target._arch != "90":
                self.skipTest("SM90-specific candidates")
            with target:
                from aitemplate.backend.cuda.conv2d.common import emit_instance_3x
                for channels in (192, 256):
                    for op_name in ("conv2d_bias_relu", "conv2d_bias_add_relu"):
                        op = getattr(ops, op_name)(stride=1, pad=1, dilate=1)
                        args = [Tensor([8, 9, 9, channels]),
                                Tensor([channels, 3, 3, channels]), Tensor([channels])]
                        if op_name.endswith("add_relu"):
                            args.append(Tensor([8, 9, 9, channels]))
                        op(*args)
                        registry.get(f"cuda.{op_name}.config")(op._attrs)
                        candidates = [k for k in op._attrs["op_instance"].values()
                                      if getattr(k, "is_3x", False)]
                        aligned = [k for k in candidates if k.tile_description.tile_shape[1] == 192]
                        self.assertEqual(bool(aligned), channels == 192)
                        self.assertTrue(any(k.tile_description.tile_shape[1] == 128 for k in candidates))
                        for kernel in aligned:
                            source = emit_instance_3x(kernel)
                            self.assertIn("cutlass::arch::Sm90", source)
                            self.assertIn("cute::_192", source)
                            self.assertNotIn("Sm100", source)


if __name__ == "__main__":
    unittest.main()
