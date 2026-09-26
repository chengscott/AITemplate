#pragma once
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
namespace ait::native_fusion {
using namespace cute;
using namespace cutlass::epilogue::fusion;
// Full-width FP16 producer reduction for the SM100 row-per-thread TMEM layout.
// The three 64-column epilogue visits cover one complete 192-column row.
// Independent partial sums shorten the floating-point dependency chain.
struct RowSquares : Sm90VisitorImpl<> {
  struct SharedStorage {};
  struct Arguments {
    float* sums;
  };
  using Params = Arguments;
  template <class P>
  static bool can_implement(P const& problem, Arguments const&) {
    auto shape = cute::append<4>(problem, 1);
    return cute::get<1>(shape) == 192 && cute::get<3>(shape) == 1;
  }
  template <class P>
  static size_t get_workspace_size(P const&, Arguments const&) {
    return 0;
  }
  template <class P>
  static auto initialize_workspace(
      P const&,
      Arguments const&,
      void*,
      cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  template <class P>
  static Params to_underlying_arguments(P const&, Arguments const& a, void*) {
    return a;
  }
  Params params;
  CUTLASS_HOST_DEVICE RowSquares(Params const& p, SharedStorage const&)
      : params(p) {}

  struct Consumer : EmptyConsumerStoreCallbacks {
    Params p;
    int row, rows;
    float ss[4] = {0.f, 0.f, 0.f, 0.f};
    CUTLASS_DEVICE Consumer(Params x, int r, int m) : p(x), row(r), rows(m) {}
    template <class A, class I, int F>
    CUTLASS_DEVICE auto visit(
        cutlass::Array<A, F> const&,
        int,
        int,
        int,
        cutlass::Array<I, F> const& value) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < F; ++i) {
        float x = float(value[i]);
        ss[i % 4] += x * x;
      }
      return value;
    }
    CUTLASS_DEVICE void end() {
      if (row < rows)
        p.sums[row] = (ss[0] + ss[1]) + (ss[2] + ss[3]);
    }
  };
  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      ConsumerStoreArgs<Args...> const& a) {
    static_assert(cute::size<0>(decltype(a.tile_shape_mnk){}) == 128);
    static_assert(cute::size<1>(decltype(a.tile_shape_mnk){}) == 192);
    return Consumer(
        params,
        cute::get<0>(a.tile_coord_mnkl) * 128 + a.thread_idx,
        cute::get<0>(a.problem_shape_mnkl));
  }
};

} // namespace ait::native_fusion
