#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ait::native_fusion::profiling {

// End each independent sample before the next sample can launch via PDL.
// Internal dependent launches inside a candidate remain enabled.
__global__ void sample_boundary() {}

inline void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
struct BufferSpec {
  int64_t count;
  int type; // FP16, E4M3, E8M0 scale, workspace bytes, or small FP16 weights.
  size_t bytes() const {
    return size_t(count) * ((type == 0 || type == 4) ? 2 : 1);
  }
};
__global__ void fill(void* ptr, int64_t count, int type, unsigned seed) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x) {
    unsigned hash = unsigned(i) * 1664525u + seed * 1013904223u;
    float value = (int(hash % 257) - 128) * (1.f / 512.f);
    if (type == 0 || type == 4)
      static_cast<__half*>(ptr)[i] =
          __float2half(value * (type == 4 ? 0.2f : 1.f));
    else if (type == 1)
      static_cast<unsigned char*>(ptr)[i] = __nv_fp8_e4m3(value).__x;
    else
      static_cast<unsigned char*>(ptr)[i] = type == 2 ? 127 : 255;
  }
}
__global__ void compare(
    __half const* a,
    __half const* b,
    int64_t count,
    int* errors) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x) {
    float x = __half2float(a[i]), y = __half2float(b[i]);
    if (!isfinite(x) || !isfinite(y) || fabsf(x - y) > .005f + .005f * fabsf(y))
      atomicAdd(errors, 1);
  }
}
inline float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// Device resources have one owner, including failure paths during validation.
struct Resources {
  cudaStream_t stream = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  std::vector<void*> allocations;
  std::vector<cudaGraph_t> graphs;
  std::vector<cudaGraphExec_t> executions;
  void* allocate(size_t bytes) {
    void* p = nullptr;
    check(cudaMalloc(&p, bytes));
    allocations.push_back(p);
    return p;
  }
  ~Resources() {
    for (auto e : executions)
      cudaGraphExecDestroy(e);
    for (auto g : graphs)
      cudaGraphDestroy(g);
    for (auto p : allocations)
      cudaFree(p);
    if (begin)
      cudaEventDestroy(begin);
    if (end)
      cudaEventDestroy(end);
    if (stream)
      cudaStreamDestroy(stream);
  }
};

template <class Specs, class Candidate>
int profile_main(
    int argc,
    char** argv,
    int count,
    Specs specs,
    std::vector<int> const& outputs,
    Candidate candidate) {
  try {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver));
    check(cudaRuntimeGetVersion(&runtime));
    if (argc == 2 && std::string(argv[1]) == "--info") {
      std::cout << "{\"name\":" << std::quoted(prop.name)
                << ",\"arch\":" << prop.major * 10 + prop.minor
                << ",\"sms\":" << prop.multiProcessorCount
                << ",\"memory\":" << prop.totalGlobalMem
                << ",\"driver\":" << driver << ",\"runtime\":" << runtime
                << ",\"nvcc\":"
                << std::quoted(
                       std::to_string(__CUDACC_VER_MAJOR__) + "." +
                       std::to_string(__CUDACC_VER_MINOR__) + "." +
                       std::to_string(__CUDACC_VER_BUILD__))
                << ",\"host_compiler\":" << std::quoted(__VERSION__) << "}\n";
      return 0;
    }
    if (argc != 3)
      throw std::runtime_error("Expected row count and RMS epsilon");
    int rows = std::stoi(argv[1]);
    float eps = std::stof(argv[2]);
    if (rows < 1 || !std::isfinite(eps) || eps <= 0)
      throw std::runtime_error("Invalid profile shape/epsilon");
    Resources resources;
    check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&resources.begin));
    check(cudaEventCreate(&resources.end));
    auto stream = resources.stream;
    auto buffers = specs(rows);
    std::vector<void*> p, reference;
    for (auto s : buffers)
      p.push_back(resources.allocate(s.bytes()));
    for (int o : outputs)
      reference.push_back(resources.allocate(buffers[o].bytes()));
    int* errors = static_cast<int*>(resources.allocate(sizeof(int)));
    std::vector<bool> valid(count, true);
    auto run = [&](int id) { return candidate(id, p, rows, eps, stream); };
    auto poison = [&]() {
      for (int o : outputs)
        check(cudaMemsetAsync(p[o], 255, buffers[o].bytes(), stream));
      for (size_t i = 0; i < buffers.size(); ++i)
        if (buffers[i].type == 3)
          check(cudaMemsetAsync(p[i], 255, buffers[i].bytes(), stream));
    };
    for (unsigned seed : {17u, 29u}) {
      for (size_t i = 0; i < buffers.size(); ++i)
        fill<<<256, 256, 0, stream>>>(
            p[i], buffers[i].count, buffers[i].type, seed + unsigned(i));
      poison();
      if (run(0))
        throw std::runtime_error("Native fallback initialization failed");
      for (size_t j = 0; j < outputs.size(); ++j)
        check(cudaMemcpyAsync(
            reference[j],
            p[outputs[j]],
            buffers[outputs[j]].bytes(),
            cudaMemcpyDeviceToDevice,
            stream));
      check(cudaStreamSynchronize(stream));
      for (int id = 0; id < count; ++id) {
        if (!valid[id])
          continue;
        poison();
        if (run(id)) {
          valid[id] = false;
          continue;
        }
        check(cudaMemsetAsync(errors, 0, sizeof(int), stream));
        for (size_t j = 0; j < outputs.size(); ++j)
          compare<<<256, 256, 0, stream>>>(
              static_cast<__half*>(p[outputs[j]]),
              static_cast<__half*>(reference[j]),
              buffers[outputs[j]].count,
              errors);
        int failures = 0;
        check(cudaMemcpyAsync(
            &failures, errors, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        valid[id] = failures == 0;
      }
    }
    if (!valid[0])
      throw std::runtime_error("Native fallback produced nonfinite outputs");
    constexpr int captures = 3, rounds = 5, replays = 25, sequence_repeats = 8;
    std::vector<std::vector<float>> samples(count * captures);
    std::vector<int> owners;
    for (int capture = 0; capture < captures; ++capture) {
      for (int index = 0; index < count; ++index) {
        int id = capture % 2 ? count - 1 - index : index;
        if (!valid[id])
          continue;
        for (int warm = 0; warm < 3; ++warm)
          if (run(id))
            throw std::runtime_error("Warmup failed");
        check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        int status = 0;
        for (int sequence = 0; sequence < sequence_repeats && !status;
             ++sequence) {
          sample_boundary<<<1, 1, 0, stream>>>();
          status = run(id);
        }
        cudaGraph_t graph = nullptr;
        auto capture_error = cudaStreamEndCapture(stream, &graph);
        if (graph)
          resources.graphs.push_back(graph);
        check(capture_error);
        if (status)
          throw std::runtime_error("Captured fused launch failed");
        cudaGraphExec_t execution = nullptr;
        check(cudaGraphInstantiate(&execution, graph, nullptr, nullptr, 0));
        resources.executions.push_back(execution);
        owners.push_back(id * captures + capture);
        poison();
        for (int warm = 0; warm < 5; ++warm)
          check(cudaGraphLaunch(execution, stream));
        check(cudaMemsetAsync(errors, 0, sizeof(int), stream));
        for (size_t j = 0; j < outputs.size(); ++j)
          compare<<<256, 256, 0, stream>>>(
              static_cast<__half*>(p[outputs[j]]),
              static_cast<__half*>(reference[j]),
              buffers[outputs[j]].count,
              errors);
        int failures = 0;
        check(cudaMemcpyAsync(
            &failures, errors, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        if (failures)
          throw std::runtime_error(
              "Captured candidate failed numerical validation");
      }
    }
    check(cudaStreamSynchronize(stream));
    for (int round = 0; round < rounds; ++round) {
      for (size_t step = 0; step < owners.size(); ++step) {
        size_t index = (step + round) % owners.size();
        if (round % 2)
          index = owners.size() - 1 - index;
        auto execution = resources.executions[index];
        check(cudaEventRecord(resources.begin, stream));
        for (int r = 0; r < replays; ++r)
          check(cudaGraphLaunch(execution, stream));
        check(cudaEventRecord(resources.end, stream));
        check(cudaEventSynchronize(resources.end));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, resources.begin, resources.end));
        samples[owners[index]].push_back(ms / (replays * sequence_repeats));
      }
    }
    std::cout << std::setprecision(9) << "{\"candidates\":[";
    for (int id = 0; id < count; ++id) {
      if (id)
        std::cout << ',';
      std::cout << "{\"id\":" << id
                << ",\"valid\":" << (valid[id] ? "true" : "false");
      if (valid[id]) {
        std::vector<float> medians;
        std::cout << ",\"capture_ms\":[";
        for (int c = 0; c < captures; ++c) {
          if (c)
            std::cout << ',';
          medians.push_back(median(samples[id * captures + c]));
          std::cout << medians.back();
        }
        std::cout << "],\"median_ms\":" << median(medians);
      }
      std::cout << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "Native fusion profiling failed: " << e.what() << '\n';
    return 1;
  }
}
} // namespace ait::native_fusion::profiling
