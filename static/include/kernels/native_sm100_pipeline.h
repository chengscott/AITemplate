#pragma once
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
namespace ait::native_fusion {
// Static single-CTA orchestration. Collective mainloops retain their TMA and
// UMMA implementations; each warp role advances the same independent tile
// stream.
template <class Base>
struct StaticGemmKernel : Base {
  using typename Base::Params;
  using Main = typename Base::CollectiveMainloop;
  using Epi = typename Base::CollectiveEpilogue;
  using MainPipe = typename Base::MainloopPipeline;
  using EpiPipe = typename Base::EpiLoadPipeline;
  using StorePipe = typename Base::EpiStorePipeline;
  using AccPipe = typename Base::AccumulatorPipeline;
  using Scheduler = typename Base::TileScheduler;
  using Cluster = typename Base::ClusterShape;
  using Tile = typename Base::TileShape;
  using CtaTile = typename Base::CtaShape_MNK;
  using Mma = typename Base::TiledMma;
  using EpiTile = typename Base::EpilogueTile;
  static_assert(cute::size(Cluster{}) == 1);
  static constexpr bool Overlap = Base::IsOverlappingAccum;
  static_assert(!Base::IsSchedDynamicPersistent);
  static_assert(Base::NumEpilogueThreads == 128);
  struct SharedStorage {
    alignas(16) typename Main::PipelineStorage main_pipe;
    alignas(16) typename Epi::PipelineStorage epi_pipe;
    alignas(16) typename AccPipe::SharedStorage acc_pipe;
    alignas(16) cutlass::arch::ClusterBarrier tmem_done;
    uint32_t tmem;
    alignas(128) typename Main::TensorStorage main;
    alignas(128) typename Epi::TensorStorage epi;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= Base::ArchTag::kSharedMemoryCapacityBytes);
  CUTLASS_DEVICE void operator()(Params const& p, char* smem) {
    using namespace cute;
    if constexpr (Overlap)
      static_assert(Main::UsesAlternatingScaleStorage);
    auto& s = *reinterpret_cast<SharedStorage*>(smem);
    const int warp = cutlass::canonical_warp_idx_sync();
    const bool epilogue = warp < 4, mma = warp == 4, load = warp == 5,
               epi_load = warp == 6;
    const bool elected = elect_one_sync();
    auto problem = append<4>(p.problem_shape, Int<1>{});
    Main main(p.mainloop, Cluster{}, 0);
    Epi epi(p.epilogue, s.epi);
    bool need_c = epi.is_producer_load_needed();
    if (load && elected)
      main.prefetch_tma_descriptors();
    if (epi_load && elected)
      epi.prefetch_tma_descriptors(p.epilogue);
    typename MainPipe::Params mp{};
    if (load)
      mp.role = MainPipe::ThreadCategory::Producer;
    if (mma)
      mp.role = MainPipe::ThreadCategory::Consumer;
    mp.is_leader = load && elected;
    mp.transaction_bytes = Main::TmaTransactionBytes;
    mp.initializing_warp = 0;
    MainPipe main_pipe(s.main_pipe, mp, Cluster{}, true_type{}, false_type{});
    typename EpiPipe::Params ep{};
    if (epi_load)
      ep.role = EpiPipe::ThreadCategory::Producer;
    if (epilogue)
      ep.role = EpiPipe::ThreadCategory::Consumer;
    ep.dst_blockid = 0;
    ep.producer_arv_count = 32;
    ep.consumer_arv_count = 128;
    ep.transaction_bytes = Epi::TmaTransactionBytes;
    ep.initializing_warp = 1;
    EpiPipe epi_pipe(s.epi_pipe, ep);
    typename StorePipe::Params sp{};
    sp.always_wait = true;
    StorePipe store_pipe(sp);
    typename AccPipe::Params ap{};
    if (mma)
      ap.role = AccPipe::ThreadCategory::Producer;
    if (epilogue)
      ap.role = AccPipe::ThreadCategory::Consumer;
    ap.producer_arv_count = 1;
    ap.consumer_arv_count = 128;
    ap.initializing_warp = 2;
    AccPipe acc_pipe(s.acc_pipe, ap, Cluster{}, true_type{}, false_type{});
    cutlass::arch::NamedBarrier tmem_ready(
        160, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    if constexpr (Overlap) {
      if (mma && elected)
        s.tmem_done.init(128);
    }
    cutlass::pipeline_init_arrive_relaxed(1);
    main_pipe.init_masks(Cluster{}, dim3(0, 0, 0));
    acc_pipe.init_masks(Cluster{}, dim3(0, 0, 0));
    auto inputs = main.load_init(problem, s.main);
    auto tm = main.template init_tmem_tensors<EpiTile, Overlap>(EpiTile{});
    Scheduler scheduler(p.scheduler);
    auto work = scheduler.initial_work_tile_info(Cluster{});
    cutlass::pipeline_init_wait(1);
    if (load) {
      cutlass::arch::wait_on_dependent_grids();
      auto state = cutlass::make_producer_start_state<MainPipe>();
      while (work.is_valid()) {
        auto coord = scheduler.work_tile_to_cta_coord(work);
        auto ki = scheduler.get_k_tile_iterator(
            work, problem, CtaTile{}, inputs.k_tiles);
        int nk = Scheduler::get_work_k_tile_count(work, problem, CtaTile{});
        auto next = main.load(main_pipe, state, inputs, coord, ki, nk);
        state = get<0>(next);
        __syncwarp();
        work = get<0>(scheduler.fetch_next_work(work));
      }
      main.load_tail(main_pipe, state);
    } else if (mma) {
      typename Base::TmemAllocator allocator;
      allocator.allocate(Base::ArchTag::kTmemCapacityColumns, &s.tmem);
      __syncwarp();
      tmem_ready.arrive();
      uint32_t ptr = s.tmem;
      main.set_tmem_offsets(tm, ptr);
      auto mi = main.mma_init(tm, s.main);
      typename MainPipe::PipelineState ms;
      auto as = cutlass::make_producer_start_state<AccPipe>();
      while (work.is_valid()) {
        auto coord = scheduler.work_tile_to_cta_coord(work);
        int nk = Scheduler::get_work_k_tile_count(work, problem, CtaTile{});
        ms = main.mma(
            make_tuple(main_pipe, acc_pipe),
            make_tuple(ms, as),
            main.slice_accumulator(tm, Overlap ? (as.phase() ^ 1) : as.index()),
            mi,
            coord,
            nk);
        acc_pipe.producer_commit(as);
        ++as;
        work = get<0>(scheduler.fetch_next_work(work));
      }
      cutlass::arch::launch_dependent_grids();
      allocator.release_allocation_lock();
      if constexpr (Overlap)
        s.tmem_done.wait(0);
      else
        acc_pipe.producer_tail(as);
      allocator.free(ptr, Base::ArchTag::kTmemCapacityColumns);
    } else if (epi_load && need_c) {
      cutlass::arch::wait_on_dependent_grids();
      auto es = cutlass::make_producer_start_state<EpiPipe>();
      auto ss = cutlass::make_producer_start_state<StorePipe>();
      while (work.is_valid()) {
        es = epi.template load<false>(
            epi_pipe,
            es,
            problem,
            CtaTile{},
            scheduler.work_tile_to_cta_coord(work),
            Tile{},
            Mma{},
            s.epi,
            false);
        work = get<0>(scheduler.fetch_next_work(work));
      }
      epi.load_tail(epi_pipe, es, store_pipe, ss);
    } else if (epilogue) {
      // Callbacks can read auxiliary vectors before waiting on TMEM.
      cutlass::arch::wait_on_dependent_grids();
      tmem_ready.arrive_and_wait();
      main.set_tmem_offsets(tm, s.tmem);
      typename AccPipe::PipelineState as;
      typename EpiPipe::PipelineState es;
      auto ss = cutlass::make_producer_start_state<StorePipe>();
      while (work.is_valid()) {
        auto acc = get<0>(
            main.slice_accumulator(tm, Overlap ? as.phase() : as.index()));
        auto next = epi.template store<Overlap>(
            epi_pipe,
            es,
            store_pipe,
            ss,
            acc_pipe,
            as,
            problem,
            CtaTile{},
            scheduler.work_tile_to_cta_coord(work),
            Tile{},
            Mma{},
            acc,
            s.epi);
        es = get<0>(next);
        ss = get<1>(next);
        as = get<2>(next);
        work = get<0>(scheduler.fetch_next_work(work));
      }
      if constexpr (Overlap)
        s.tmem_done.arrive();
      epi.store_tail(epi_pipe, es, store_pipe, ss, CtaTile{});
    }
  }
};
} // namespace ait::native_fusion
