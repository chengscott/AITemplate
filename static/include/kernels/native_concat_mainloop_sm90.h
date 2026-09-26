/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Adapted to load contiguous gate/up halves into one accumulator tile.
#pragma once
#include "cutlass/gemm/collective/collective_builder.hpp"
namespace ait::native_fusion {
// Load the separate gate/up weight halves into adjacent halves of one MMA tile.
// The half-width TMA views retain the full mainloop's pipeline-stage stride.
template <class Base, class Half>
struct ConcatMainloop90 : Base {
  static_assert(cute::size(typename Base::DispatchPolicy::ClusterShape{}) == 1);
  static_assert(
      cute::size<1>(typename Base::TileShape{}) ==
      2 * cute::size<1>(typename Half::TileShape{}));
  static_assert(576 % cute::size<1>(typename Half::TileShape{}) == 0);
  using typename Base::Arguments;
  using typename Base::MainloopPipeline;
  using typename Base::PipelineState;
  using typename Base::TensorStorage;
  struct Params : Base::Params {
    typename Half::Params::TMA_B weight;
  };
  template <class PS>
  static Params to_underlying_arguments(
      PS const& shape,
      Arguments const& a,
      void* ws) {
    typename Half::Arguments ha{a.ptr_A, a.dA, a.ptr_B, a.dB};
    return {
        Base::to_underlying_arguments(shape, a, ws),
        Half::to_underlying_arguments(shape, ha, ws).tma_load_b};
  }
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const& p) {
    cute::prefetch_tma_descriptor(p.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.weight.get_tma_descriptor());
  }
  template <class PS>
  CUTLASS_DEVICE auto load_init(PS const& problem, Params const& p) const {
    using namespace cute;
    auto old = Base::load_init(problem, p);
    auto [M, N, K, L] = problem;
    auto b = local_tile(
        p.weight.get_tma_tensor(make_shape(N, K, L)),
        typename Half::TileShape{},
        make_coord(_, _, _),
        Step<Underscore, _1, _1>{});
    return make_tuple(get<0>(old), b);
  }
  template <class A, class B, class C, class K>
  CUTLASS_DEVICE void load(
      Params const& p,
      MainloopPipeline pipe,
      PipelineState state,
      cute::tuple<A, B> const& in,
      C const& coord,
      K ki,
      int count,
      int tid,
      uint32_t rank,
      TensorStorage& shared) {
    using namespace cute;
    if (!elect_one_sync())
      return;
    auto sa = make_tensor(
        make_smem_ptr(shared.smem_A.data()), typename Base::SmemLayoutA{});
    auto half = typename Half::SmemLayoutB{};
    auto layout = composition(
        half.layout_a(),
        half.offset(),
        make_layout(
            shape(half),
            replace<2>(
                stride(half.layout_b()),
                stride<2>(typename Base::SmemLayoutB{}.layout_b()))));
    constexpr int offset = size<1>(typename Half::TileShape{}) *
        size<2>(typename Half::TileShape{});
    auto sg = make_tensor(make_smem_ptr(shared.smem_B.data()), layout);
    auto su = make_tensor(make_smem_ptr(shared.smem_B.data() + offset), layout);
    auto ta = p.tma_load_a.get_slice(0);
    auto tb = p.weight.get_slice(0);
    auto ga = ta.partition_S(get<0>(in)(_, _, get<0>(coord), _, get<3>(coord)));
    auto as = ta.partition_D(sa);
    auto gg = tb.partition_S(get<1>(in)(_, _, get<1>(coord), _, get<3>(coord)));
    auto gs = tb.partition_D(sg);
    auto gu = tb.partition_S(
        get<1>(in)(
            _,
            _,
            get<1>(coord) + 576 / size<1>(typename Half::TileShape{}),
            _,
            get<3>(coord)));
    auto us = tb.partition_D(su);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < count; ++k) {
      pipe.producer_acquire(state);
      auto* barrier = pipe.producer_get_barrier(state);
      int s = state.index();
      copy(p.tma_load_a.with(*barrier, 0), ga(_, _, _, *ki), as(_, _, _, s));
      copy(p.weight.with(*barrier, 0), gg(_, _, _, *ki), gs(_, _, _, s));
      copy(p.weight.with(*barrier, 0), gu(_, _, _, *ki), us(_, _, _, s));
      ++ki;
      ++state;
    }
  }
};
} // namespace ait::native_fusion
