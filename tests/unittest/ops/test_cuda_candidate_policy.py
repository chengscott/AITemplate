# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from aitemplate.backend.cuda import utils


class CUDACandidatePolicyTestCase(unittest.TestCase):
    def test_hopper_fallback_policy(self):
        native = NS(gemm_kind="3x")
        fallback = NS(gemm_kind="2x")
        conv = NS(is_3x=True)
        library = NS(OperationKind=NS(Gemm="gemm", Conv2d="conv"),
                     GemmKind=NS(Universal3x="3x"))
        for nested in (False, True):
            for force in (False, True):
                for setting in (None, "0", "1"):
                    with self.subTest(nested=nested, force=force, setting=setting):
                        def manifest_factory(args):
                            gemms = {"mixed": [native, fallback], "fallback": [fallback]}
                            convs = {"conv": [conv]}
                            return NS(operations={
                                "gemm": {90: gemms} if nested else gemms,
                                "conv": {90: convs} if nested else convs,
                            })
                        lib = NS(library=library,
                                 manifest=NS(Manifest=manifest_factory),
                                 generator=NS(GenerateSM90=mock.Mock(), GenerateSM80=mock.Mock()),
                                 extra_operation=NS(GenerateSM80=mock.Mock()))
                        env = {} if setting is None else {"AIT_SM90_ALLOW_SM80_GEMM": setting}
                        with mock.patch.dict(os.environ, env, clear=True), \
                             mock.patch.dict(sys.modules, {"cutlass_lib": lib}), \
                             mock.patch.object(utils, "_generate_sm90_conv3x_f16_f32acc"):
                            result = utils.gen_ops("90", "13.0", True, force)
                        self.assertEqual(result["conv"], {"conv": [conv]})
                        expected = {"mixed": [native]} if setting == "0" else {
                            "mixed": [native, fallback], "fallback": [fallback]}
                        self.assertEqual(result["gemm"], expected)

    def test_blackwell_conv_profiler_pool(self):
        def conv(cluster, inst, schedule="1sm", native=True):
            return NS(is_3x=native, kernel_schedule=schedule,
                      tile_description=NS(cluster_shape=cluster,
                                          math_instruction=NS(instruction_shape=inst)))
        single = conv([1, 1, 1], [128, 128, 16])
        multicast = conv([2, 2, 1], [64, 64, 16])
        fallback = conv([1, 1, 1], [16, 8, 16], native=False)
        invalid = [conv([2, 1, 1], [128, 128, 16]),
                   conv([1, 1, 1], [128, 128, 16], "2sm"),
                   conv([0, 1, 1], [64, 64, 16])]
        lib = NS(library=NS(OperationKind=NS(Conv2d="conv")))
        for nested in (False, True):
            with self.subTest(nested=nested):
                configs = {"candidates": [single, multicast, fallback] + invalid}
                manifest = NS(operations={"conv": {100: configs} if nested else configs})
                utils._filter_sm100_conv_ops(lib, manifest)
                self.assertEqual(configs["candidates"], [single, multicast, fallback])


if __name__ == "__main__":
    unittest.main()
