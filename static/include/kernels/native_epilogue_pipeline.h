#pragma once
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
namespace ait::native_fusion {
// Keep residual loads independent of asynchronous output stores.
template <class T>
struct IndependentEpilogueStorage;
template <int C, int D, int F, bool R, bool Delay, class... A>
struct IndependentEpilogueStorage<
    cutlass::epilogue::collective::CollectiveEpilogue<
        cutlass::epilogue::Sm90TmaWarpSpecialized<C, D, F, R, Delay>,
        A...>> {
  using Type = cutlass::epilogue::collective::CollectiveEpilogue<
      cutlass::epilogue::Sm90TmaWarpSpecialized<C, D, F, false, Delay>,
      A...>;
};
template <int C, int D, int F, bool R, bool Delay, class... A>
struct IndependentEpilogueStorage<
    cutlass::epilogue::collective::CollectiveEpilogue<
        cutlass::epilogue::Sm100TmaWarpSpecialized<C, D, F, R, Delay>,
        A...>> {
  using Type = cutlass::epilogue::collective::CollectiveEpilogue<
      cutlass::epilogue::Sm100TmaWarpSpecialized<C, D, F, false, Delay>,
      A...>;
};
} // namespace ait::native_fusion
