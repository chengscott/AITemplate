# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import unittest

from aitemplate.backend import registry
from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target


class SM100ChannelTilesTestCase(unittest.TestCase):
    def setUp(self):
        self.target = detect_target(use_fp16_acc=False)
        if self.target.name() != "cuda" or self.target._arch != "100":
            self.skipTest("SM100-specific kernel candidates")

    def test_gemm_names_and_shape_scope(self):
        with self.target:
            for channels in (192, 576, 1152, 256):
                op = ops.gemm_rcr_bias()
                op(
                    Tensor([81, 384], dtype="float16"),
                    Tensor([channels, 384], dtype="float16"),
                    Tensor([channels], dtype="float16"),
                )
                registry.get("cuda.gemm_rcr_bias.config")(op._attrs)
                candidates = list(op._attrs["op_instance"].values())
                aligned = [
                    kernel for kernel in candidates
                    if list(kernel.tile_description.tile_shape) == [128, 192, 64]
                ]
                self.assertEqual(bool(aligned), channels % 192 == 0)
                for kernel in aligned:
                    # A deepcopy must invalidate CUTLASS cached_property names.
                    # Otherwise two different C++ types get the same name and
                    # the profiler cannot compile both candidates together.
                    self.assertIn("_128x192x64_", kernel.procedural_name())
                    self.assertEqual(kernel.get_collective_tile_shape(), (128, 192, 64))
                original = [
                    kernel for kernel in candidates
                    if list(kernel.tile_description.tile_shape) == [128, 256, 64]
                    and list(kernel.tile_description.cluster_shape) == [1, 1, 1]
                    and kernel.arch == 100
                ]
                self.assertTrue(original)
                for kernel in original:
                    self.assertIn("_128x256x64_", kernel.procedural_name())

    def test_residual_gemm_candidate_names(self):
        with self.target:
            for op_name in ("gemm_rcr_bias_add", "gemm_rcr_bias_add_relu"):
                for channels in (192, 384, 256):
                    op = getattr(ops, op_name)()
                    op(
                        Tensor([81, 192], dtype="float16"),
                        Tensor([channels, 192], dtype="float16"),
                        Tensor([channels], dtype="float16"),
                        Tensor([81, channels], dtype="float16"),
                    )
                    registry.get(f"cuda.{op_name}.config")(op._attrs)
                    aligned = [
                        kernel for kernel in op._attrs["op_instance"].values()
                        if list(kernel.tile_description.tile_shape) == [128, 192, 64]
                    ]
                    self.assertEqual(len(aligned), int(channels % 192 == 0))
                    for kernel in aligned:
                        self.assertIn("_128x192x64_", kernel.procedural_name())

    def test_convolution_keeps_original_and_aligned_tiles(self):
        with self.target:
            for channels in (192, 256):
                op = ops.conv2d_bias_relu(stride=1, pad=1, dilate=1)
                op(
                    Tensor([8, 9, 9, channels], dtype="float16"),
                    Tensor([channels, 3, 3, channels], dtype="float16"),
                    Tensor([channels], dtype="float16"),
                )
                registry.get("cuda.conv2d_bias_relu.config")(op._attrs)
                tiles = {
                    tuple(kernel.tile_description.tile_shape)
                    for kernel in op._attrs["op_instance"].values()
                    if getattr(kernel, "is_3x", False)
                }
                self.assertIn((128, 128, 64), tiles)
                self.assertEqual((128, 192, 64) in tiles, channels == 192)
                if channels == 192:
                    self.assertIn((64, 64, 64), tiles)
                    self.assertIn((128, 64, 64), tiles)
                    from aitemplate.backend.cuda.conv2d.common import emit_instance_3x

                    paired = [
                        kernel for kernel in op._attrs["op_instance"].values()
                        if getattr(kernel, "is_3x", False)
                        and "2Sm" in str(kernel.kernel_schedule)
                    ]
                    self.assertEqual(len(paired), 1)
                    source = emit_instance_3x(paired[0])
                    self.assertIn("cute::_256, cute::_192", source)
                    self.assertNotIn("cute::_512", source)


if __name__ == "__main__":
    unittest.main()
