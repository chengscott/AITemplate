#pragma once

// Enable CUTLASS's device-side dependency instructions before its headers are
// parsed. Setting the launch attribute alone leaves these helpers as no-ops.
#ifndef CUTLASS_ENABLE_GDC_FOR_SM90
#define CUTLASS_ENABLE_GDC_FOR_SM90
#endif
#ifndef CUTLASS_ENABLE_GDC_FOR_SM100
#define CUTLASS_ENABLE_GDC_FOR_SM100
#endif
#include "cutlass/arch/grid_dependency_control.h"

#if defined(__CUDA_ARCH__) && \
    ((defined(__CUDA_ARCH_FEAT_SM90_ALL) && __CUDA_ARCH__ == 900) || \
     (defined(__CUDA_ARCH_FEAT_SM100_ALL) && __CUDA_ARCH__ == 1000))
static_assert(
    cutlass::arch::IsGdcGloballyEnabled,
    "Include native_gdc.h before CUTLASS headers to enable dependent launches");
#endif
