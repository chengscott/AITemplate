# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import pathlib
import shutil
import subprocess
import tempfile
import unittest

import torch

from aitemplate.backend.cuda.gemm_universal import common, common_bias_broadcast


class GemmProfilerWorkspaceTestCase(unittest.TestCase):
    def test_candidate_workspace_lifetime(self):
        nvcc = shutil.which("nvcc")
        if nvcc is None or not torch.cuda.is_available():
            self.skipTest("Requires nvcc and CUDA")
        root = pathlib.Path(__file__).resolve().parents[3]
        cutlass = root / "3rdparty/cutlass"
        body = common.EXEC_TEMPLATE.render(
            instance="GemmInstance", is_profiler=True, profile_with_cuda_graph=True,
            problem_args="", problem_args_cutlass_3x="",
        )
        # Exercise allocation and launch without depending on a particular GEMM tile.
        fake_gemm = r"""
template<int Id> struct WorkspaceOp {
  using ElementAccumulator = float;
  struct GemmKernel {};
  struct Arguments {};
  uint8_t* ptr = nullptr;
  bool reject = false;
  size_t get_workspace_size(Arguments const&) { return 32 * 1024 * 1024; }
  cutlass::Status can_implement(Arguments const&) {
    if (reject) throw std::runtime_error("rejected candidate");
    return cutlass::Status::kSuccess;
  }
  cutlass::Status initialize(Arguments const&, void* p, cudaStream_t) {
    ptr = static_cast<uint8_t*>(p);
    return cutlass::Status::kSuccess;
  }
  cutlass::Status operator()(cudaStream_t stream) {
    if (cudaMemsetAsync(ptr, 42, 32 * 1024 * 1024, stream) != cudaSuccess)
      throw std::runtime_error("workspace launch failed");
    return cutlass::Status::kSuccess;
  }
};
"""
        op_func = common_bias_broadcast.SRC_TEMPLATE.render(
            is_profiler=True, profile_with_cuda_graph=True,
            instances=fake_gemm, function_name="gemm", input_ndims=0,
            weight_ndims=0, exec_paths=body,
        )
        call = common_bias_broadcast.FUNC_CALL_TEMPLATE.render(
            is_profiler=True, profile_with_cuda_graph=True, func_name="gemm",
            a_ptr="nullptr", b_ptr="nullptr", c_ptr="nullptr",
            bias_ptr="reinterpret_cast<void*>(1)", d0_ptr="reinterpret_cast<void*>(1)",
        )
        cases = r"""
  auto check = ProfilerGraphResources::check;
  check(cudaFree(nullptr));
  size_t before, after, total;
  check(cudaMemGetInfo(&before, &total));
  WorkspaceOp<0> a;
  WorkspaceOp<1> b;
  // Different specializations must not retain separate allocations after benchmarking.
  benchmark_gemm(a, "a", memory_pool.get(), nullptr, nullptr);
  benchmark_gemm(b, "b", memory_pool.get(), nullptr, nullptr);
  check(cudaMemGetInfo(&after, &total));
  if (before > after + 4 * 1024 * 1024)
    throw std::runtime_error("successful candidates retained workspace");
  a.reject = true;
  bool rejected = false;
  try { benchmark_gemm(a, "reject", memory_pool.get(), nullptr, nullptr); }
  catch (std::runtime_error const&) { rejected = true; }
  if (!rejected) throw std::runtime_error("expected rejection");
  check(cudaMemGetInfo(&after, &total));
  if (before > after + 4 * 1024 * 1024)
    throw std::runtime_error("rejected candidate retained workspace");
  // Unwind during capture, then verify a subsequent candidate still works.
  try {
    ProfilerGraphResources resources;
    check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
    resources.workspace.reset(32 * 1024 * 1024);
    check(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal));
    check(cudaMemsetAsync(resources.workspace.get(), 0, 32 * 1024 * 1024, resources.stream));
    throw std::runtime_error("abort capture");
  } catch (std::runtime_error const&) {}
  benchmark_gemm(b, "after_abort", memory_pool.get(), nullptr, nullptr);
  check(cudaMemGetInfo(&after, &total));
  if (before > after + 4 * 1024 * 1024)
    throw std::runtime_error("aborted capture retained workspace");
"""
        source = "#include <cstddef>\n" + common.PROFILER_TEMPLATE.render(
            profile_with_cuda_graph=True, op_func=op_func, function_name="gemm",
            elem_type="float", input_ndims=0, weight_ndims=0, output_ndims=0,
            func_call=call, benchmark_instances=cases,
        )
        with tempfile.TemporaryDirectory() as work:
            src = pathlib.Path(work) / "workspace.cu"
            exe = pathlib.Path(work) / "workspace"
            src.write_text(source)
            build = subprocess.run([
                nvcc, "-std=c++17", "-O0", str(src), "-o", str(exe),
                "-I" + str(cutlass / "include"),
                "-I" + str(cutlass / "tools/util/include"),
            ], capture_output=True, text=True, timeout=180)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
