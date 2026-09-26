"""Cache isolation and exact-shape dispatch for native fusion profiling."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aitemplate.backend.profiler_cache import ProfileCacheDB
from aitemplate.backend.cuda.gemm_universal import (
    native_fused_gemm_profiler as profiler,
)
from aitemplate.compiler.base import ExecItem
from aitemplate.frontend import IntVar, Tensor


class NativeFusionProfilerTest(unittest.TestCase):
    def test_architecture_defaults_and_explicit_overrides(self):
        from aitemplate.backend.cuda.gemm_universal.native_fused_gemm import enabled
        from aitemplate.backend.cuda.gemm_universal.gemm_rms_swiglu import gen_function

        for arch in (80, 90, 100):
            with patch.object(profiler.Target, "current", return_value=SimpleNamespace(_arch=str(arch))):
                with patch.dict("os.environ", {}, clear=True):
                    self.assertTrue(enabled())
                    self.assertEqual(profiler._native_enabled(), arch == 80)
                    with patch.dict("sys.modules", {"quack": None}):
                        code = gen_function(dict(self.attrs(), name="default_probe"))
                    header = "native_rms_swiglu_sm80.h" if arch == 80 else "native_rms_swiglu.h"
                    self.assertIn(header, code)
                for backend in ("auto", "cutlass", "quack"):
                    with patch.dict("os.environ", {"AIT_FUSED_GEMM_BACKEND": backend}, clear=True):
                        native = backend in ("auto", "cutlass")
                        self.assertEqual(enabled(), native)
                        for tuning in ("0", "1"):
                            with patch.dict("os.environ", {"AIT_NATIVE_FUSION_TUNING": tuning}):
                                self.assertEqual(profiler._native_enabled(), native and tuning == "1")
                with patch.dict("os.environ", {"AIT_FUSED_GEMM_BACKEND": "invalid"}, clear=True):
                    with self.assertRaises(ValueError):
                        enabled()

    def attrs(self):
        return dict(
            op="gemm_rms_swiglu",
            joint=True,
            eps=1e-6,
            profile_rows=None,
            inputs=[Tensor([IntVar([1, 32]), 81, 192], dtype="float16")],
        )

    def test_dynamic_rows_and_exact_dispatch(self):
        attrs = self.attrs()
        self.assertEqual(profiler.profile_rows(attrs), [81, 162, 324, 648, 1296, 2592])
        attrs["profile_rows"] = (243, 2592)
        self.assertEqual(profiler.profile_rows(attrs), [243, 2592])
        attrs["exec_path"] = {"243": ExecItem("243", "rows == 243", "m64n64p0e0c00")}
        code = profiler.function(attrs, 90, "void test()")
        self.assertIn("if(rows==243)", code)
        self.assertIn("run_tuned<90,true,64,64,false,false>", code)
        self.assertIn("swiglu::run<90,true>", code)
        attrs["profile_rows"] = (0,)
        with self.assertRaises(ValueError):
            profiler.profile_rows(attrs)

    def test_connected_pair_context_and_buffers(self):
        from aitemplate.compiler import ops
        from aitemplate.backend.cuda.gemm_universal.native_fusion_profiler_source import (
            render,
        )

        b = IntVar([1, 32])
        x = Tensor([b, 81, 192])
        r = Tensor([b, 81, 192])
        ffn = ops.gemm_rms_swiglu(eps=3e-4)
        updated, hidden = ffn(x, Tensor([1152, 192]), r, Tensor([192, 192]))
        boundary = ops.gemm_rms_relu_gemm()
        boundary(
            hidden,
            Tensor([192, 576]),
            updated,
            Tensor([192]),
            Tensor([384, 192]),
            Tensor([b, 81, 384]),
        )
        context = profiler.profile_context(boundary._attrs)
        self.assertEqual(context, profiler.profile_context(ffn._attrs))
        self.assertEqual(context["ffn_eps"], 3e-4)
        attrs = dict(boundary._attrs, native_profile_context=context)
        choices = profiler.candidates("gemm_rms_relu_gemm", 100)
        self.assertEqual(len(choices), 13)
        self.assertEqual(
            {(c["pi"], c["ci"]) for c in choices.values() if c},
            {(False, False), (False, True), (True, False), (True, True)},
        )
        code = render(attrs, 100, choices)
        self.assertIn("swiglu::run<100,true>", code)
        self.assertIn("boundary::run<100,128,false,128,false,true>", code)
        self.assertIn("0.0003f", code)
        boundary._attrs["inputs"][2] = r
        self.assertIsNone(profiler.profile_context(boundary._attrs))

    def test_disabled_backend_keeps_fallback(self):
        with patch.dict("os.environ", {"AIT_FUSED_GEMM_BACKEND": "auto"}):
            attrs = self.attrs()
            self.assertEqual(profiler.gen_profiler(attrs, "unused"), [])
            self.assertEqual(attrs["exec_path"], {})

    def test_relative_workdir_and_ampere_header_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "cutlass/examples/45_dual_gemm/threadblock/dual_mma.h"
            header.parent.mkdir(parents=True)
            header.write_text("// original\n")
            target = SimpleNamespace(
                _arch="80",
                static_files_path=str(root / "static"),
                template_path=lambda: str(root / "cutlass"),
                compile_options=lambda: "-O3",
                use_dummy_profiling_results=lambda: False,
            )
            with patch.object(profiler.Target, "current", return_value=target), patch.dict(
                "os.environ",
                {"AIT_FUSED_GEMM_BACKEND": "cutlass", "AIT_NATIVE_FUSION_TUNING": "1"},
            ):
                attrs = self.attrs()
                source, binary = profiler.gen_profiler(attrs, os.path.relpath(root / "first"))[0]
                self.assertTrue(Path(source).is_absolute())
                self.assertTrue(Path(source).is_file())
                self.assertTrue(Path(binary).is_absolute())
                fingerprint = attrs["native_profile_fingerprint"]
                profiler.gen_profiler(attrs, os.path.relpath(root / "second"))
                self.assertEqual(fingerprint, attrs["native_profile_fingerprint"])
                header.write_text("// changed\n")
                profiler.gen_profiler(attrs, os.path.relpath(root / "second"))
                self.assertNotEqual(fingerprint, attrs["native_profile_fingerprint"])

    def test_cache_hit_and_epsilon_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ProfileCacheDB("cuda", path=str(Path(directory) / "cache.db"))
            binary = Path(directory) / "profiler"
            binary.write_bytes(b"test profiler binary")

            class Target:
                _arch = "90"

                def dev_select_flag(self):
                    return "CUDA_VISIBLE_DEVICES"

                def force_profile(self):
                    return False

                def query_profile_cache(self, kind, args):
                    self_kind = kind
                    assert self_kind == "native_fusion"
                    return cache.query_native_fusion(args)

                def insert_profile_cache(self, kind, args):
                    assert kind == "native_fusion"
                    cache.insert_native_fusion(args)

            calls = []

            def run(command, **kwargs):
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
                calls.append(command)
                result = (
                    {"arch": 90, "name": "H200", "sms": 132}
                    if command[-1] == "--info"
                    else {
                        "candidates": [
                            {
                                "id": 0,
                                "valid": True,
                                "median_ms": 0.020,
                                "capture_ms": [0.020, 0.020, 0.020],
                            },
                            {
                                "id": 1,
                                "valid": True,
                                "median_ms": 0.010,
                                "capture_ms": [0.010, 0.010, 0.010],
                            },
                        ]
                    }
                )

                class Result:
                    stdout = json.dumps(result)

                return Result()

            attrs = self.attrs()
            attrs["native_profiler"] = str(binary)
            attrs["native_profile_fingerprint"] = "source-v1"

            def reset():
                attrs["exec_path"] = {"81": ExecItem("81", "rows == 81", "")}

            with patch.object(
                profiler.Target, "current", return_value=Target()
            ), patch.object(profiler.subprocess, "run", side_effect=run), patch.dict(
                "os.environ",
                {"AIT_FUSED_GEMM_BACKEND": "cutlass", "AIT_NATIVE_FUSION_TUNING": "1"},
            ):
                reset()
                profiler.profile(attrs, [2])
                self.assertEqual(len(calls), 2)
                self.assertEqual(attrs["exec_path"]["81"].algo, "m64n64p0e0c00")
                reset()
                profiler.profile(attrs, [2])
                self.assertEqual(len(calls), 3)
                attrs["eps"] = 3e-4
                reset()
                profiler.profile(attrs, [2])
                self.assertEqual(len(calls), 5)
                copied_binary = Path(directory) / "another-build"
                copied_binary.write_bytes(binary.read_bytes())
                attrs["native_profiler"] = str(copied_binary)
                reset()
                profiler.profile(attrs, [2])
                self.assertEqual(len(calls), 6)
                attrs["native_profile_fingerprint"] = "source-v2"
                reset()
                profiler.profile(attrs, [2])
                self.assertEqual(len(calls), 8)
                attrs["native_profile_fingerprint"] = "missing-source"
                reset()
                with patch.object(
                    profiler.environ, "force_profiler_cache", return_value=True
                ):
                    with self.assertRaisesRegex(RuntimeError, "Missing native fusion"):
                        profiler.profile(attrs, [2])
                self.assertEqual(len(calls), 9)

    def test_tuning_is_opt_in(self):
        with patch.dict(
            "os.environ", {"AIT_FUSED_GEMM_BACKEND": "cutlass"}, clear=True
        ):
            self.assertFalse(profiler._native_enabled())

    def test_candidates_and_disabled_tuning(self):
        with self.assertRaises(NotImplementedError):
            profiler.candidates("gemm_rms_relu_gemm", 80)
        ampere = profiler.candidates("gemm_rms_swiglu", 80)
        self.assertEqual(len(ampere), 17)
        attrs = self.attrs()
        attrs["exec_path"] = {"243": ExecItem("243", "rows == 243", "m128n64s3i1")}
        code = profiler.function(attrs, 80, "void test()")
        self.assertIn('"native_rms_swiglu_sm80.h"', code)
        self.assertIn("sm80::run<128,64,3,true,true>", code)
        self.assertNotIn("launch_sm90", code)
        joint = profiler.candidates("gemm_rms_swiglu", 90, True)
        plain = profiler.candidates("gemm_rms_swiglu", 90, False)
        self.assertEqual(len(joint), 25)
        self.assertEqual(len(plain), 13)
        self.assertTrue(all(c is None or c["pc"] == 0 for c in plain.values()))
        with patch.dict(
            "os.environ",
            {"AIT_FUSED_GEMM_BACKEND": "cutlass", "AIT_NATIVE_FUSION_TUNING": "0"},
        ):
            attrs = self.attrs()
            self.assertEqual(profiler.gen_profiler(attrs, "unused"), [])
            self.assertEqual(attrs["exec_path"], {})


if __name__ == "__main__":
    unittest.main()
