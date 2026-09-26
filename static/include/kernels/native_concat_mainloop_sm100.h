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
struct ConcatMainloop : Base {
  static_assert(cute::size(typename Base::DispatchPolicy::ClusterShape{}) == 1);
  static_assert(
      cute::size<1>(typename Base::TileShape{}) ==
      2 * cute::size<1>(typename Half::TileShape{}));
  static_assert(576 % cute::size<1>(typename Half::TileShape{}) == 0);
  using typename Base::Arguments;
  using ClusterShape = typename Base::DispatchPolicy::ClusterShape;
  using typename Base::MainloopPipeline;
  using typename Base::MainloopPipelineState;
  using typename Base::TensorStorage;
  using TmaHalf = typename Half::Params::TMA_B;
  struct Params {
    typename Base::Params base;
    TmaHalf weight;
  };
  TmaHalf const* weight;
  CUTLASS_DEVICE ConcatMainloop(Params const& p, ClusterShape c, uint32_t rank)
      : Base(p.base, c, rank), weight(&p.weight) {}
  template <class PS>
  static Params to_underlying_arguments(
      PS const& shape,
      Arguments const& a,
      void* ws,
      cutlass::KernelHardwareInfo const& hw = {}) {
    typename Half::Arguments ha{a.ptr_A, a.dA, a.ptr_B, a.dB};
    return {
        Base::to_underlying_arguments(shape, a, ws, hw),
        Half::to_underlying_arguments(shape, ha, ws, hw).tma_load_b};
  }
  CUTLASS_DEVICE void prefetch_tma_descriptors() {
    cute::prefetch_tma_descriptor(
        this->observed_tma_load_a_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(weight->get_tma_descriptor());
  }
  template <class O, class G, class S>
  struct Inputs {
    decltype(std::declval<O>().k_tiles) k_tiles;
    O original;
    G global;
    S gate, up;
  };
  template <class PS>
  CUTLASS_DEVICE auto load_init(PS const& problem, TensorStorage& shared)
      const {
    using namespace cute;
    auto old = Base::load_init(problem, shared);
    auto [M, N, K, L] = problem;
    auto gb = local_tile(
        weight->get_tma_tensor(make_shape(N, K, L)),
        typename Half::TileShape{},
        make_coord(_, _, _),
        Step<Underscore, _1, _1>{});
    auto part = typename Half::TiledMma{}.get_slice(0).partition_B(gb);
    auto half = typename Half::SmemLayoutB{};
    auto layout = composition(
        half.layout_a(),
        half.offset(),
        make_layout(
            shape(half),
            replace<3>(
                stride(half.layout_b()),
                stride<3>(typename Base::SmemLayoutB{}.layout_b()))));
    constexpr int offset = size<1>(typename Half::TileShape{}) *
        size<2>(typename Half::TileShape{});
    auto sg = make_tensor(make_smem_ptr(shared.smem_B.begin()), layout);
    auto su =
        make_tensor(make_smem_ptr(shared.smem_B.begin() + offset), layout);
    auto [global, gate] = tma_partition(
        *weight,
        Int<0>{},
        Layout<_1>{},
        group_modes<0, 3>(sg),
        group_modes<0, 3>(part));
    auto [unused, up] = tma_partition(
        *weight,
        Int<0>{},
        Layout<_1>{},
        group_modes<0, 3>(su),
        group_modes<0, 3>(part));
    return Inputs<decltype(old), decltype(global), decltype(gate)>{
        old.k_tiles, old, global, gate, up};
  }
  template <class I, class C, class K>
  CUTLASS_DEVICE auto load(
      MainloopPipeline pipe,
      MainloopPipelineState state,
      I const& in,
      C const& coord,
      K ki,
      int count) {
    using namespace cute;
    auto a = in.original.tAgA_mkl(_, get<0>(coord), _, get<3>(coord));
    auto gate = in.global(_, get<1>(coord), _, get<3>(coord));
    auto up = in.global(
        _,
        get<1>(coord) + 576 / size<1>(typename Half::TileShape{}),
        _,
        get<3>(coord));
    auto token = pipe.producer_try_acquire(state);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < count; ++k) {
      pipe.producer_acquire(state, token);
      auto* barrier = pipe.producer_get_barrier(state);
      int stage = state.index();
      ++state;
      token = pipe.producer_try_acquire(state);
      if (elect_one_sync()) {
        copy(
            this->observed_tma_load_a_->with(*barrier, 0),
            a(_, *ki),
            in.original.tAsA(_, stage));
        copy(weight->with(*barrier, 0), gate(_, *ki), in.gate(_, stage));
        copy(weight->with(*barrier, 0), up(_, *ki), in.up(_, stage));
      }
      ++ki;
    }
    return make_tuple(state, ki);
  }
};
} // namespace ait::native_fusion
