"""Backend variants must not reuse legacy cache entries or profiler executables."""

import os
import pathlib
import tempfile
import unittest
from collections import OrderedDict
from hashlib import sha1
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aitemplate.backend.cuda.gemm_universal import common, common_bias_broadcast
from aitemplate.backend.target import Target
from aitemplate.compiler import ops
from aitemplate.compiler.base import ExecItem
from aitemplate.compiler.ops.gemm_universal.gemm_common import GemmProfilerPostprocessingDelegate


class GemmProfilerNamespaceTestCase(unittest.TestCase):
    def test_variant_roundtrip_and_legacy_compatibility(self):
        key = "M == 256 && N == 128 && K == 256"
        tensor = SimpleNamespace(element=SimpleNamespace(value=2), layout=SimpleNamespace(value=1))
        kernel = SimpleNamespace(A=tensor, B=tensor, C=tensor,
                                 epilogue_functor=SimpleNamespace(value=1),
                                 accumulator_type=lambda: SimpleNamespace(value=3))
        op = ops.gemm_rcr_bias_relu()
        op._attrs.update(name="gemm", op_instance=OrderedDict(kernel=kernel),
                         exec_path=OrderedDict([(key, ExecItem(key, key, ""))]))
        cache = {}
        records = []

        def insert(kind, record):
            records.append(record)
            cache[record["exec_entry_sha1"]] = (record["algo"], record["workspace"], record["split_k"])

        target = SimpleNamespace(
            _arch="90", use_dummy_profiling_results=lambda: False, force_profile=lambda: False,
            get_profile_cache_version=lambda kind: 3,
            query_profile_cache=lambda kind, query: cache.get(query["exec_entry_sha1"]),
            insert_profile_cache=insert,
        )

        def record_result(algo):
            delegate = GemmProfilerPostprocessingDelegate()
            delegate.add_instance(((algo, .1, 0), op._attrs, "profiler", key, 1))
            delegate.postprocess_results()

        with patch.object(Target, "current", return_value=target), patch(
            "aitemplate.compiler.ops.gemm_universal.gemm_common.environ.force_profiler_cache",
            return_value=False,
        ):
            original_filename = op._get_profiler_filename()
            self.assertEqual(original_filename, "gemm_rcr_bias_relu_" + sha1(b"kernel").hexdigest() + "_3")
            record_result("legacy")
            self.assertIn(sha1(key.encode()).hexdigest(), cache)
            self.assertEqual(records[-1]["exec_entry"], key)

            op._attrs["profile_cache_namespace"] = "evt_v1"
            self.assertNotEqual(original_filename, op._get_profiler_filename())
            self.assertTrue(op._should_build_profiler([key], op._attrs["op_instance"]))
            record_result("evt")
            self.assertEqual(records[-1]["exec_entry"], "evt_v1:" + key)
            self.assertFalse(op._should_build_profiler([key], op._attrs["op_instance"]))
            self.assertEqual(op._attrs["exec_path"][key].algo, "evt")

            op._attrs["exec_path"][key].algo = ""
            runner = Mock()
            op._profile_single_workload("unused", key, runner, force_cache=True)
            self.assertEqual(op._attrs["exec_path"][key].algo, "evt")
            runner.push.assert_not_called()

            with patch.dict(os.environ, {"AIT_SM90_ALLOW_SM80_GEMM": "0"}):
                self.assertTrue(op._should_build_profiler([key], op._attrs["op_instance"]))
                record_result("native")
                self.assertEqual(records[-1]["exec_entry"], "evt_v1:sm90_native_only:" + key)
                self.assertFalse(op._should_build_profiler([key], op._attrs["op_instance"]))
                self.assertEqual(op._attrs["exec_path"][key].algo, "native")
            self.assertFalse(op._should_build_profiler([key], op._attrs["op_instance"]))
            self.assertEqual(op._attrs["exec_path"][key].algo, "evt")

            del op._attrs["profile_cache_namespace"]
            self.assertFalse(op._should_build_profiler([key], op._attrs["op_instance"]))
            self.assertEqual(op._attrs["exec_path"][key].algo, "legacy")
            self.assertEqual(original_filename, op._get_profiler_filename())

            # An executable and result from the old workspace-lifetime implementation
            # must not suppress rebuilding or retuning under the current namespace.
            op._attrs["profile_cache_namespace"] = "sm90_residual_evt_graph_v2"
            record_result("old_workspace")
            old_filename = op._get_profiler_filename()
            with tempfile.TemporaryDirectory() as work:
                prefix = pathlib.Path(work) / "profiler" / op._attrs["op"]
                prefix.mkdir(parents=True)
                (prefix / old_filename).touch()
                op._attrs["profile_cache_namespace"] = (
                    common_bias_broadcast._SM90_RESIDUAL_PROFILE_NAMESPACE
                )
                new_filename = op._get_profiler_filename()
                self.assertNotEqual(old_filename, new_filename)
                self.assertTrue(op._should_build_profiler([key], op._attrs["op_instance"]))
                file_pairs = []
                common.add_profiler(file_pairs, work, op._attrs["op"], new_filename, "new profiler")
                self.assertEqual(len(file_pairs), 1)
                self.assertEqual(pathlib.Path(file_pairs[0][1]).name, new_filename)
                self.assertEqual(pathlib.Path(file_pairs[0][0]).read_text(), "new profiler")


if __name__ == "__main__":
    unittest.main()
