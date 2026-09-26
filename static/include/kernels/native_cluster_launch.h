#pragma once

#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

namespace ait::native_fusion {

// Preserve dependent launch while explicitly requesting a single-CTA cluster.
// Hopper graph replay can differ from an otherwise identical ordinary launch.
template <class Gemm>
int launch_sm90_cluster(Gemm const& gemm, cudaStream_t stream) {
  using Kernel = typename Gemm::GemmKernel;
  static_assert(Kernel::ArchTag::kMinComputeCapability == 90);
  auto const& params = gemm.params();
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  attributes[1].id = cudaLaunchAttributeClusterDimension;
  attributes[1].val.clusterDim = {1, 1, 1};
  cudaLaunchConfig_t config{};
  config.gridDim = Kernel::get_grid_shape(params);
  config.blockDim = Kernel::get_block_shape();
  config.dynamicSmemBytes = Kernel::SharedStorageSize;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 2;
  auto error = cudaLaunchKernelEx(&config, cutlass::device_kernel<Kernel>, params);
  if (error == cudaSuccess) {
    error = cudaGetLastError();
  }
  return int(error == cudaSuccess ? cutlass::Status::kSuccess
                                 : cutlass::Status::kErrorInternal);
}

} // namespace ait::native_fusion
