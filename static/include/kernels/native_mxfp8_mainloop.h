#pragma once
namespace ait::native_fusion {
// Alternate full-width accumulator halves. The next tile's scale factors reuse
// the previous tile's first columns only after its epilogue has loaded them.
template <class Base>
struct BlockScaledMainloop : Base {
  using Base::Base;
  static constexpr bool UsesAlternatingScaleStorage = true;
  using typename Base::MainloopPipelineState;
  template <class EpiTile, bool Overlap = false>
  CUTLASS_DEVICE static auto init_tmem_tensors(EpiTile epi) {
    if constexpr (!Base::IsOverlappingAccum)
      return Base::template init_tmem_tensors<EpiTile, Overlap>(epi);
    else {
      using namespace cute;
      static_assert(
          size<0>(typename Base::TileShape{}) == 128 &&
          size<1>(typename Base::TileShape{}) == 256);
      typename Base::TiledMma mma;
      auto acc = cutlass::detail::make_sm100_accumulator<2, false>(
          mma, Base::partition_accumulator_shape(), epi);
      auto old = Base::template init_tmem_tensors<EpiTile, Overlap>(epi);
      typename Base::template TmemStorage<
          decltype(acc),
          decltype(old.tCtSFA),
          decltype(old.tCtSFB)>
          tm;
      tm.accumulators = acc;
      tm.tCtSFA = old.tCtSFA;
      tm.tCtSFB = old.tCtSFB;
      return tm;
    }
  }
  template <class T>
  CUTLASS_DEVICE static void set_tmem_offsets(T& tm, uint32_t ptr) {
    if constexpr (!Base::IsOverlappingAccum)
      Base::set_tmem_offsets(tm, ptr);
    else {
      tm.accumulators.data() = ptr;
      tm.tCtSFA.data() = ptr + 256;
      tm.tCtSFB.data() =
          ptr + 256 + cutlass::detail::find_tmem_tensor_col_offset(tm.tCtSFA);
    }
  }
  template <class Pipes, class States, class Acc, class Inputs, class Coord>
  CUTLASS_DEVICE MainloopPipelineState
  mma(Pipes pipes,
      States states,
      Acc const& acc,
      Inputs const& in,
      Coord coord,
      int nk) {
    if constexpr (!Base::IsOverlappingAccum)
      return Base::mma(pipes, states, acc, in, coord, nk);
    else {
      using namespace cute;
      auto [mp, ap] = pipes;
      auto [ms, as] = states;
      auto [mma, a, b, sfa, sfb, copya, srca, dsta, copyb, srcb, dstb] = in;
      auto output = get<0>(acc);
      uint32_t scale_ptr = output.data().get() ^ 256u;
      uint32_t offset = cutlass::detail::find_tmem_tensor_col_offset(sfa);
      sfa.data() = scale_ptr;
      sfb.data() = scale_ptr + offset;
      dsta.data() = scale_ptr;
      dstb.data() = scale_ptr + offset;
      auto token = mp.consumer_try_wait(ms, nk <= 0);
      ap.producer_acquire(as);
      mma.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k = 0; k < nk; ++k) {
        mp.consumer_wait(ms, token);
        int stage = ms.index();
        if (elect_one_sync()) {
          copy(copya, srca(_, _, _, _, stage), dsta);
          copy(copyb, srcb(_, _, _, _, stage), dstb);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<2>(a); ++j) {
          cute::gemm(
              mma.with(mma.accumulate_, sfa(_, _, j), sfb(_, _, j)),
              a(_, _, j, stage),
              b(_, _, j, stage),
              output);
          mma.accumulate_ = UMMA::ScaleOut::One;
        }
        mp.consumer_release(ms);
        ++ms;
        token = mp.consumer_try_wait(ms, k + 1 >= nk);
      }
      return ms;
    }
  }
};
} // namespace ait::native_fusion
