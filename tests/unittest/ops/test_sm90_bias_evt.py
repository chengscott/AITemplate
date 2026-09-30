# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from aitemplate.backend import registry
from aitemplate.backend.target import Target
from aitemplate.backend.cuda.gemm_universal import common_bias_activation, common_bias_broadcast
from aitemplate.backend.cuda.gemm_universal.common import has_tma_epilogue
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target


class SM90BiasEVTTestCase(unittest.TestCase):
    def test_default_policy(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            Target, "current", return_value=SimpleNamespace(_arch="90")
        ):
            self.assertTrue(common_bias_activation.activation_use_evt())
            self.assertIn("90", common_bias_broadcast._residual_evt_arches())
            with mock.patch.dict(os.environ, {"AIT_SM90_ALLOW_SM80_GEMM": "0"}):
                self.assertTrue(common_bias_activation.activation_use_evt())
                self.assertEqual(common_bias_broadcast._residual_evt_arches(), ("90", "100"))

    def test_activation_and_residual_evts(self):
        self._test_evts(("relu", "sigmoid", "tanh", "swish", "add", "add_relu", "mul"))

    def test_other_activation_evts(self):
        self._test_evts(("gelu", "fast_gelu", "hardswish"))

    def _test_evts(self, kinds):
        with mock.patch.dict(os.environ, {"AIT_FORCE_CUTLASS_SM90_KERNELS": "1"}):
            target = detect_target(allow_cutlass_sm90=True, force_cutlass_sm90=True)
        if target.name() != "cuda" or target._arch != "90":
            self.skipTest("Requires Hopper")
        torch.manual_seed(123)
        settings = {"AIT_SM90_ALLOW_SM80_GEMM": "0",
                    "AIT_FORCE_CUTLASS_SM90_KERNELS": "1"}
        with tempfile.TemporaryDirectory() as work, mock.patch.dict(os.environ, settings):
            for kind in kinds:
                with self.subTest(kind=kind):
                    m = IntVar([1, 256], name="m")
                    shapes = [[m, 256], [128, 256], [128]]
                    residual = kind in ("add", "add_relu", "mul")
                    if residual:
                        shapes.append([m, 128])
                    inputs = [Tensor(s, dtype="float16", name=f"x{i}", is_input=True)
                              for i, s in enumerate(shapes)]
                    op_name = "gemm_rcr_bias_" + kind
                    op = getattr(ops, op_name)()
                    y = op(*inputs)
                    y._attrs.update(name="y", is_output=True)
                    key = f"cuda.{op_name}.config"
                    configure = registry.get(key)

                    def evt_only(attrs, *args, **kwargs):
                        configure(attrs, *args, **kwargs)
                        expected_namespace = (
                            "sm90_residual_evt_graph_v3" if residual else "sm90_bias_activation_evt_v1"
                        )
                        self.assertEqual(attrs["profile_cache_namespace"], expected_namespace)
                        attrs["op_instance"] = {
                            name: kernel for name, kernel in attrs["op_instance"].items()
                            if has_tma_epilogue(kernel)
                        }
                        self.assertTrue(attrs["op_instance"])

                    # A fallback kernel must not hide broken EVT code generation.
                    with mock.patch.dict(registry.BACKEND_FUNCTIONS, {key: evt_only}), \
                         mock.patch.dict(os.environ, {"FORCE_PROFILE": "1"}):
                        model = compile_model(y, target, work, kind)
                    for batch in (1, 7, 256):
                        tensors = [torch.randn(batch, 256, device="cuda", dtype=torch.float16) * .2,
                                   torch.randn(128, 256, device="cuda", dtype=torch.float16) * .2,
                                   torch.randn(128, device="cuda", dtype=torch.float16) * .2]
                        ref = tensors[0].float() @ tensors[1].float().T + tensors[2].float()
                        if residual:
                            tensors.append(torch.randn(batch, 128, device="cuda", dtype=torch.float16) * .2)
                            ref = ref * tensors[3].float() if kind == "mul" else ref + tensors[3].float()
                        if kind in ("relu", "add_relu"):
                            ref = ref.relu()
                        elif kind == "sigmoid":
                            ref = ref.sigmoid()
                        elif kind == "tanh":
                            ref = ref.tanh()
                        elif kind == "swish":
                            ref = torch.nn.functional.silu(ref)
                        elif kind in ("gelu", "fast_gelu"):
                            ref = torch.nn.functional.gelu(
                                ref, approximate="tanh" if kind == "fast_gelu" else "none"
                            )
                        elif kind == "hardswish":
                            ref = torch.nn.functional.hardswish(ref)
                        out = torch.empty(batch, 128, device="cuda", dtype=torch.float16)
                        model.run_with_tensors(tensors, [out])
                        torch.testing.assert_close(out, ref.half(), atol=.002, rtol=.002)


if __name__ == "__main__":
    unittest.main()
