"""Partial profile-cache hits must not suppress later dynamic GEMM buckets."""

import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aitemplate.backend.target import Target
from aitemplate.compiler import ops
from aitemplate.compiler.base import ExecItem
from aitemplate.compiler.base import DynamicProfileStrategy
from aitemplate.frontend import IntVar, Tensor


class GemmProfilerBucketTest(unittest.TestCase):
    def test_large_shape_policy_keeps_small_buckets(self):
        op = ops.gemm_rcr()
        op(Tensor([IntVar([81, 165888]), 192]), Tensor([384, 192]))
        with patch.dict("os.environ", {"AIT_GEMM_M_BUCKETS": "4", "AIT_GEMM_M_BUCKET_POLICY": "log"}):
            op._extract_exec_path(DynamicProfileStrategy.MAX)
            original = set(op._attrs["exec_path"])
        with patch.dict("os.environ", {"AIT_GEMM_M_BUCKETS": "4", "AIT_GEMM_M_BUCKET_POLICY": "large"}):
            op._extract_exec_path(DynamicProfileStrategy.MAX)
            expanded = set(op._attrs["exec_path"])
        self.assertTrue(original <= expanded)
        for m in (82944, 103680, 124416, 145152):
            self.assertIn(f"M == {m} && N == 384 && K == 192", expanded)
        with patch.dict("os.environ", {"AIT_GEMM_M_BUCKETS": "4", "AIT_GEMM_M_BUCKET_POLICY": "invalid"}):
            with self.assertRaises(ValueError):
                op._extract_exec_path(DynamicProfileStrategy.MAX)

    def test_backend_bucket_ratio_refines_dynamic_shapes(self):
        op = ops.gemm_rcr()
        op(Tensor([IntVar([81, 165888]), 192]), Tensor([384, 192]))
        op._attrs["max_profile_bucket_ratio"] = 2.0
        with patch.dict("os.environ", {"AIT_GEMM_M_BUCKETS": "4", "AIT_GEMM_M_BUCKET_POLICY": "log"}):
            op._extract_exec_path(DynamicProfileStrategy.MAX)
            keys = list(op._attrs["exec_path"])
            self.assertEqual(keys, [f"M == {81 * 2**i} && N == 384 && K == 192" for i in range(12)])
            for ratio in (1, 0, float("nan"), float("inf")):
                op._attrs["max_profile_bucket_ratio"] = ratio
                with self.assertRaises(ValueError):
                    op._extract_exec_path(DynamicProfileStrategy.MAX)

    def test_partial_cache_hits_profile_every_missing_bucket(self):
        for cached in ({0}, {1}, {0, 2}, {0, 1, 2, 3}, set()):
            with self.subTest(cached=cached):
                op = ops.gemm_rcr()
                keys = [f"M == {m} && N == 384 && K == 192" for m in (81, 648, 10368, 165888)]
                op._attrs.update(
                    name="bucketed_gemm", op_instance={},
                    exec_path=OrderedDict(
                        (key, ExecItem(key, key, "cached_kernel" if i in cached else ""))
                        for i, key in enumerate(keys)
                    ),
                )
                target = SimpleNamespace(use_dummy_profiling_results=lambda: False)
                runner = object()
                with patch.object(Target, "current", return_value=target), patch(
                    "aitemplate.compiler.ops.gemm_universal.gemm_common.environ.force_profiler_cache",
                    return_value=False,
                ), patch.object(op, "_profile_single_workload", new=Mock()) as profile:
                    op.profile(runner, workdir="build")
                self.assertEqual(
                    [call.args[1] for call in profile.call_args_list],
                    [key for i, key in enumerate(keys) if i not in cached],
                )
                for i in cached:
                    self.assertEqual(op._attrs["exec_path"][keys[i]].algo, "cached_kernel")


if __name__ == "__main__":
    unittest.main()
