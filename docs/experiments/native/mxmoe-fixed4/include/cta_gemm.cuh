#pragma once
#include "mm_tile.cuh"

namespace tiled_gemm {

/// different pipeline
template <typename TileConfig, typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void cta_gemm_multistage_v2(  //
    TileA& tile_a, TileB& tile_b, TileC& tile_c,         //
    uint8_t* smem,                                       //
    int K, int lane, int warp_x, int warp_y, int warp_z  //
) {
  using TA = typename TileA::TA;
  using TB = typename TileB::TB;
  static_assert(std::is_same_v<TA, half> && std::is_same_v<TB, half>, "Only half is supported for now");

  using FragA = typename TileA::FragA;
  using FragB = typename TileB::FragB;
  using FragC = typename TileC::FragC;

  constexpr int BK          = TileConfig::BK;
  constexpr int STAGE       = TileConfig::STAGE;
  constexpr int WARP_K      = TileConfig::WARP_K;
  constexpr int WARP_ITER_K = TileConfig::WARP_ITER_K;

  constexpr int SMEM_SIZE_A = TileA::VEC_PER_ROW * TileA::TILE_ROW;
  constexpr int SMEM_SIZE_B = TileB::VEC_PER_ROW * TileB::TILE_ROW;

  // allocate register frag
  FragC frag_c{};
  FragA frag_a[WARP_ITER_K];
  FragB frag_b[WARP_ITER_K];

  int main_loop = K / BK;

  uint4* smem_base_a = reinterpret_cast<uint4*>(smem);
  uint4* smem_base_b = reinterpret_cast<uint4*>(smem + STAGE * SMEM_SIZE_A * 16);

  int smem_ld_stage = 0;
  int smem_st_stage = 0;

  uint4* smem_ld_iter_a = smem_base_a;
  uint4* smem_ld_iter_b = smem_base_b;
  uint4* smem_st_iter_a = smem_base_a + smem_st_stage * SMEM_SIZE_A;
  uint4* smem_st_iter_b = smem_base_b + smem_st_stage * SMEM_SIZE_B;

  auto load_smem = [&]() {
    tile_a.load_smem(smem_st_iter_a, main_loop > 0);
    tile_b.load_smem(smem_st_iter_b, main_loop > 0);
    // advance gmem pointer of tileAB
    tile_a.advance_gmem();
    tile_b.advance_gmem();
    // advance smem pointer of tileAB
    smem_st_stage  = (smem_st_stage + 1) % STAGE;
    smem_st_iter_a = smem_base_a + smem_st_stage * SMEM_SIZE_A;
    smem_st_iter_b = smem_base_b + smem_st_stage * SMEM_SIZE_B;
  };

  auto load_frag = [&](int k) {
    tile_a.load_frag(k, frag_a[k], smem_ld_iter_a);
    tile_b.load_frag(k, frag_b[k], smem_ld_iter_b);
  };

  // 1. prologue
#pragma unroll
  for (auto i = 0; i < STAGE - 1; i++) {
    load_smem();
    commit_cp_async_group();
    main_loop--;
  }
  wait_cp_async_group<STAGE - 2>();
  __syncthreads();
  load_frag(0);

  // 2. Mainloop
#pragma unroll 1
  for (; main_loop > (-STAGE + 1); main_loop--) {
#pragma unroll
    for (int k = 0; k < WARP_ITER_K; ++k) {
      int pre_frag_iter = (k + 1) % WARP_ITER_K;
      if (k == WARP_ITER_K - 1) {
        smem_ld_stage  = (smem_ld_stage + 1) % STAGE;
        smem_ld_iter_a = smem_base_a + smem_ld_stage * SMEM_SIZE_A;
        smem_ld_iter_b = smem_base_b + smem_ld_stage * SMEM_SIZE_B;
        tile_a.reset_frag_offset();
        tile_b.reset_frag_offset();
        wait_cp_async_group<STAGE - 2>();
        __syncthreads();
      }

      load_frag(pre_frag_iter);

      if (k == 0) {
        load_smem();
        commit_cp_async_group();
      }

      TileConfig::mma_compute(frag_c, frag_a[k], frag_b[k]);
    }
  }

  // 3. write `tilec` to global memory
  if constexpr (WARP_K > 1) {
    tile_c.reduce_slicek(frag_c, smem, lane, warp_x, warp_y, warp_z);
  }
  tile_c.write_frag_to_gmem(frag_c, lane, warp_x, warp_y, warp_z, smem);
}

/// matrix B is quantized into [2 or 4 or 8] bits
/// smem: STAGE * (TileA + QTileB + ScaleB)
/// reg : WARP_ITER_K * (FragA + FragB + ScaleB); FragC
template <typename TileConfig, typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void cta_gemm_multistage_qb_v2(  //
    TileA& tile_a, TileB& tile_b, TileC& tile_c,            //
    uint8_t* smem,                                          //
    const half* scale_zp,                                   //
    half* smem_scale_zp,                                    //
    int N, int K,                                           //
    int lane, int warp_x, int warp_y, int warp_z, int tidx  //
) {
  using FragA = typename TileA::FragA;
  using FragB = typename TileB::FragB;
  using FragC = typename TileC::FragC;

  using QFragB = typename TileB::QFragB;
  using QScale = typename TileB::QScale;

  constexpr int BK          = TileConfig::BK;
  constexpr int STAGE       = TileConfig::STAGE;
  constexpr int WARP_K      = TileConfig::WARP_K;
  constexpr int WARP_ITER_K = TileConfig::WARP_ITER_K;

  constexpr int SMEM_SIZE_A = TileA::VEC_PER_ROW * TileA::TILE_ROW;
  constexpr int SMEM_SIZE_B = TileB::VEC_PER_ROW * TileB::TILE_ROW;

  // for group quant
  constexpr int GSIZE           = TileB::GSIZE;
  constexpr int G2S_GROUP_FETCH = cu_cdiv(GSIZE, BK);
  // constexpr int CTA_SCALE_SIZE  = TileB::CTA_SCALE_SIZE;
  constexpr int G2S_SCALE_SIZE = TileB::G2S_SCALE_SIZE;

  static_assert(GSIZE % (TileConfig::mmaK * WARP_K) == 0 || GSIZE == -1, "GSIZE must be multiple of mmaK");
  static_assert(GSIZE >= BK || GSIZE == -1, "GSIZE >= BK is required for now");

  // allocate register frag
  FragC frag_c{};
  FragA frag_a[WARP_ITER_K];
  FragB frag_b[WARP_ITER_K];
  QFragB qfrag_b[WARP_ITER_K];
  QScale r_sb[GSIZE != -1 ? WARP_ITER_K : 1];

  int main_loop  = K / TileConfig::BK;
  int mma_iter_k = 0;

  uint4* smem_base_a = reinterpret_cast<uint4*>(smem);
  uint4* smem_base_b = reinterpret_cast<uint4*>(smem + STAGE * SMEM_SIZE_A * 16);

  int smem_ld_stage     = 0;
  int smem_st_stage     = 0;
  uint4* smem_ld_iter_a = smem_base_a;
  uint4* smem_ld_iter_b = smem_base_b;
  uint4* smem_st_iter_a = smem_base_a;
  uint4* smem_st_iter_b = smem_base_b;

  int pre_ld_tile      = 0;
  int pre_group        = 0;
  int cur_group        = 0;
  half* smem_st_iter_s = smem_scale_zp;
  half* smem_ld_iter_s = smem_scale_zp;

  const half* gmem_scale_zp = scale_zp;
  const int gmem_scale_step = (TileB::SYM_QUANT ? 1 : 2) * N * cu_cdiv(BK, GSIZE);

  auto load_smem = [&]() {
    tile_a.load_smem(smem_st_iter_a, main_loop > 0);
    tile_b.load_smem(smem_st_iter_b, main_loop > 0);
    if constexpr (GSIZE != -1) {
      bool scale_pred = (pre_ld_tile % G2S_GROUP_FETCH == 0) && (main_loop > 0);
      // tile_b.load_scale_zp_smem_align(smem_st_iter_s, gmem_scale_zp, tidx, scale_pred);
      tile_b.load_scale_zp_smem(smem_st_iter_s, gmem_scale_zp, tidx, scale_pred);
      if (scale_pred) {
        // advance gmem pointer of scale
        gmem_scale_zp += gmem_scale_step;
        // advance smem pointer of scale
        pre_group      = (pre_group + 1) % STAGE;
        smem_st_iter_s = smem_scale_zp + pre_group * G2S_SCALE_SIZE;
      }
    }
    // advance gmem pointer of tileAB
    tile_a.advance_gmem();
    tile_b.advance_gmem();
    // advance smem pointer of tileAB
    smem_st_stage  = (smem_st_stage + 1) % STAGE;
    smem_st_iter_a = smem_base_a + smem_st_stage * SMEM_SIZE_A;
    smem_st_iter_b = smem_base_b + smem_st_stage * SMEM_SIZE_B;
    pre_ld_tile++;
  };

  auto load_frag = [&](int k) {
    tile_a.load_frag(k, frag_a[k], smem_ld_iter_a);
    tile_b.load_frag(k, qfrag_b[k], smem_ld_iter_b);

    if constexpr (GSIZE != -1) tile_b.load_frag_scale(r_sb[k], smem_ld_iter_s, lane, warp_x);

    // // handle the case: GROUP_SIZE < BK
    // if constexpr (G2S_SCALE_SIZE > CTA_SCALE_SIZE) {
    //   if (mma_iter_k % GSIZE == 0) {
    //     smem_ld_iter_s += CTA_SCALE_SIZE;
    //   }
    // }
    mma_iter_k += TileConfig::mmaK * WARP_K;
  };

  // 1. prologue
  // if constexpr (GSIZE == -1) tile_b.load_scale_zp_smem_align(smem_scale_zp, gmem_scale_zp, tidx, true);
  if constexpr (GSIZE == -1) tile_b.load_scale_zp_smem(smem_scale_zp, gmem_scale_zp, tidx, true);
#pragma unroll
  for (auto i = 0; i < STAGE - 1; i++) {
    load_smem();
    commit_cp_async_group();
    main_loop--;
  }
  wait_cp_async_group<STAGE - 2>();
  __syncthreads();

  load_frag(0);
  if constexpr (GSIZE == -1) tile_b.load_frag_scale(r_sb[0], smem_scale_zp, lane, warp_x);
  TileB::dequant_frag(frag_b[0], qfrag_b[0], r_sb[0]);

  // if (lane == 0 && warp_x == 0 && warp_y == 0 && warp_z == 0) {
  //   printf("smem_scale_zp:\n");
  //   for (auto i = 0; i < TileB::G2S_SCALE_SIZE; i++) {
  //     printf("%-9.3f, ", float(smem_scale_zp[i]));
  //   }
  //   printf("\n");
  // }
  // print_frag<TileConfig, 16, 16>(frag_a[0][0], 0, 0, 0, 0, 0, "frag_a");
  // print_frag<TileConfig, 16, 8, uint16_t>(qfrag_b[0][0], 0, 0, 0, 0, 0, "qfrag_b");
  // print_frag<TileConfig>(frag_b[0][0], 0, 0, 0, 0, 0, "frag_b");

  // 2. Mainloop
#pragma unroll 1
  for (; main_loop > (-STAGE + 1); main_loop--) {
    load_smem();
    commit_cp_async_group();
#pragma unroll
    for (int k = 0; k < WARP_ITER_K; k++) {
      int pre_frag_iter = (k + 1) % WARP_ITER_K;

      if (k == WARP_ITER_K - 1) {
        // switch pre-load stage
        smem_ld_stage  = (smem_ld_stage + 1) % STAGE;
        smem_ld_iter_a = smem_base_a + smem_ld_stage * SMEM_SIZE_A;
        smem_ld_iter_b = smem_base_b + smem_ld_stage * SMEM_SIZE_B;
        if constexpr (GSIZE != -1) {
          if (mma_iter_k % GSIZE == 0) {
            cur_group      = (cur_group + 1) % STAGE;
            smem_ld_iter_s = smem_scale_zp + cur_group * G2S_SCALE_SIZE;
          }
        }
        tile_a.reset_frag_offset();
        tile_b.reset_frag_offset();
        wait_cp_async_group<STAGE - 2>();
        __syncthreads();
      }

      // pre-load frag
      load_frag(pre_frag_iter);

      TileConfig::mma_compute(frag_c, frag_a[k], frag_b[k]);

      // pre-dequant
      if constexpr (GSIZE == -1) {
        TileB::dequant_frag(frag_b[pre_frag_iter], qfrag_b[pre_frag_iter], r_sb[0]);
      } else {
        TileB::dequant_frag(frag_b[pre_frag_iter], qfrag_b[pre_frag_iter], r_sb[pre_frag_iter]);
      }
    }
  }

  // 3. write `tilec` to global memory
  if constexpr (WARP_K > 1) {
    tile_c.reduce_slicek(frag_c, smem, lane, warp_x, warp_y, warp_z);
  }
  tile_c.write_frag_to_gmem(frag_c, lane, warp_x, warp_y, warp_z, smem);
}

/// matrix B is quantized into [2 or 4 or 8] bits
/// smem: STAGE * (TileA + QTileB + ScaleB)
/// reg : WARP_ITER_K * (FragA + FragB + ScaleB); FragC
template <typename TileConfig, typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void cta_gemm_wxa16g128(         //
    TileA& tile_a, TileB& tile_b, TileC& tile_c,            //
    uint8_t* smem,                                          //
    const half* scale_zp,                                   //
    half* smem_scale_zp,                                    //
    int N, int K,                                           //
    int lane, int warp_x, int warp_y, int warp_z, int tidx  //
) {
  using FragA = typename TileA::FragA;
  using FragB = typename TileB::FragB;
  using FragC = typename TileC::FragC;

  using QFragB = typename TileB::QFragB;
  using QScale = typename TileB::QScale;

  constexpr int BK          = TileConfig::BK;
  constexpr int STAGE       = TileConfig::STAGE;
  constexpr int WARP_K      = TileConfig::WARP_K;
  constexpr int WARP_ITER_K = TileConfig::WARP_ITER_K;

  constexpr int SMEM_SIZE_A = TileA::VEC_PER_ROW * TileA::TILE_ROW;
  constexpr int SMEM_SIZE_B = TileB::VEC_PER_ROW * TileB::TILE_ROW;

  // for group quant
  constexpr int GSIZE          = TileB::GSIZE;
  constexpr int G2S_SCALE_SIZE = TileB::G2S_SCALE_SIZE;

  static_assert(GSIZE % (TileConfig::mmaK * WARP_K) == 0 || GSIZE == -1, "GSIZE must be multiple of mmaK");
  static_assert(GSIZE == BK || GSIZE == 128, "GSIZE == BK is required for now");

  // allocate register frag
  FragC frag_c{};
  FragA frag_a[WARP_ITER_K];
  FragB frag_b[WARP_ITER_K];
  QFragB qfrag_b[WARP_ITER_K];
  QScale r_sb[WARP_ITER_K];

  int main_loop  = K / TileConfig::BK;
  // int mma_iter_k = 0;

  uint4* smem_base_a = reinterpret_cast<uint4*>(smem);
  uint4* smem_base_b = reinterpret_cast<uint4*>(smem + STAGE * SMEM_SIZE_A * 16);

  int smem_ld_stage     = 0;
  int smem_st_stage     = 0;
  uint4* smem_ld_iter_a = smem_base_a;
  uint4* smem_ld_iter_b = smem_base_b;
  uint4* smem_st_iter_a = smem_base_a;
  uint4* smem_st_iter_b = smem_base_b;

  half* smem_st_iter_s = smem_scale_zp;
  half* smem_ld_iter_s = smem_scale_zp;

  const half* gmem_scale_zp = scale_zp;
  const int gmem_scale_step = (TileB::SYM_QUANT ? 1 : 2) * N * cu_cdiv(BK, GSIZE);

  auto load_smem = [&]() {
    tile_a.load_smem(smem_st_iter_a, main_loop > 0);
    tile_b.load_smem(smem_st_iter_b, main_loop > 0);
    // tile_b.load_scale_zp_smem_align(smem_st_iter_s, gmem_scale_zp, tidx, main_loop > 0);
    tile_b.load_scale_zp_smem(smem_st_iter_s, gmem_scale_zp, tidx, main_loop > 0);

    // advance gmem pointer of `tileAB` and `scale`
    gmem_scale_zp += gmem_scale_step;
    tile_a.advance_gmem();
    tile_b.advance_gmem();

    // advance smem pointer of `tileAB` and `scale`
    smem_st_stage  = (smem_st_stage + 1) % STAGE;
    smem_st_iter_a = smem_base_a + smem_st_stage * SMEM_SIZE_A;
    smem_st_iter_b = smem_base_b + smem_st_stage * SMEM_SIZE_B;
    smem_st_iter_s = smem_scale_zp + smem_st_stage * G2S_SCALE_SIZE;
  };

  auto load_frag = [&](int k) {
    tile_a.load_frag(k, frag_a[k], smem_ld_iter_a);
    tile_b.load_frag(k, qfrag_b[k], smem_ld_iter_b);
    tile_b.load_frag_scale(r_sb[k], smem_ld_iter_s, lane, warp_x);
  };

  // 1. prologue
#pragma unroll
  for (auto i = 0; i < STAGE - 1; i++) {
    load_smem();
    commit_cp_async_group();
    main_loop--;
  }
  wait_cp_async_group<STAGE - 2>();
  __syncthreads();

  load_frag(0);
  TileB::dequant_frag(frag_b[0], qfrag_b[0], r_sb[0]);

  // 2. Mainloop
#pragma unroll 1
  for (; main_loop > (-STAGE + 1); main_loop--) {
    load_smem();
    commit_cp_async_group();
#pragma unroll
    for (int k = 0; k < WARP_ITER_K; k++) {
      int pre_frag_iter = (k + 1) % WARP_ITER_K;

      if (k == WARP_ITER_K - 1) {
        // switch pre-load stage
        smem_ld_stage  = (smem_ld_stage + 1) % STAGE;
        smem_ld_iter_a = smem_base_a + smem_ld_stage * SMEM_SIZE_A;
        smem_ld_iter_b = smem_base_b + smem_ld_stage * SMEM_SIZE_B;
        smem_ld_iter_s = smem_scale_zp + smem_ld_stage * G2S_SCALE_SIZE;
        tile_a.reset_frag_offset();
        tile_b.reset_frag_offset();
        wait_cp_async_group<STAGE - 2>();
        __syncthreads();
      }

      // pre-load frag
      load_frag(pre_frag_iter);

      TileConfig::mma_compute(frag_c, frag_a[k], frag_b[k]);

      // pre-dequant
      TileB::dequant_frag(frag_b[pre_frag_iter], qfrag_b[pre_frag_iter], r_sb[pre_frag_iter]);
    }
  }

  // 3. write `tilec` to global memory
  if constexpr (WARP_K > 1) {
    tile_c.reduce_slicek(frag_c, smem, lane, warp_x, warp_y, warp_z);
  }
  tile_c.write_frag_to_gmem(frag_c, lane, warp_x, warp_y, warp_z, smem);
}

template <typename TileConfig, typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void cta_gemm_multistage_qab_v2(  //
    TileA& tile_a, TileB& tile_b, TileC& tile_c,             //
    uint8_t* smem,                                           //
    const half* scale_a, const half* scale_b,                //
    half* smem_scale_a, half* smem_scale_b,                  //
    int M, int N, int K,                                     //
    int lane, int warp_x, int warp_y, int warp_z, int tidx   //
) {
  using FragA = typename TileA::FragA;
  using FragB = typename TileB::FragB;
  using FragC = typename TileC::FragC;

  using T_ACC = float;

  constexpr int BK          = TileConfig::BK;
  constexpr int STAGE       = TileConfig::STAGE;
  constexpr int WARP_K      = TileConfig::WARP_K;
  constexpr int WARP_ITER_K = TileConfig::WARP_ITER_K;

  constexpr int WARP_FRAG_M = TileConfig::WARP_FRAG_M;
  constexpr int WARP_FRAG_N = TileConfig::WARP_FRAG_N;

  constexpr int SMEM_SIZE_A = TileA::VEC_PER_ROW * TileA::TILE_ROW;
  constexpr int SMEM_SIZE_B = TileB::VEC_PER_ROW * TileB::TILE_ROW;

  // for group quant
  constexpr int GSIZE           = TileB::GSIZE;
  constexpr int G2S_GROUP_FETCH = (GSIZE + BK - 1) / BK;
  // constexpr int CTA_SCALE_SIZE_A = TileA::CTA_SCALE_SIZE;
  // constexpr int CTA_SCALE_SIZE_B = TileB::CTA_SCALE_SIZE;
  constexpr int G2S_SCALE_SIZE_A = TileA::G2S_SCALE_SIZE;
  constexpr int G2S_SCALE_SIZE_B = TileB::G2S_SCALE_SIZE;
  constexpr int MMA_STEP         = TileConfig::mmaK * WARP_K;

  static_assert(TileB::GSIZE == TileA::GSIZE, "GSIZE of TileA and TileB must be the same");
  static_assert(GSIZE % TileConfig::mmaK == 0 || GSIZE == -1, "GSIZE must be multiple of mmaK");
  static_assert(GSIZE >= BK || GSIZE == -1, "GSIZE >= BK is required for now");
  static_assert(WARP_K == 1, "WARP_K must be 1");

  // allocate register frag
  FragC frag_c{};
  FragA frag_a[WARP_ITER_K];
  FragB frag_b[WARP_ITER_K];
  half r_sa[WARP_FRAG_M * 2];
  half r_sb[WARP_FRAG_N * 2];
  T_ACC frag_c_out[WARP_FRAG_M * WARP_FRAG_N][TileConfig::FRAG_C_TH]{};

  int main_loop  = K / BK;
  int mma_iter_k = 0;

  uint4* smem_base_a = reinterpret_cast<uint4*>(smem);
  uint4* smem_base_b = reinterpret_cast<uint4*>(smem + STAGE * SMEM_SIZE_A * 16);

  int smem_ld_stage     = 0;
  int smem_st_stage     = 0;
  uint4* smem_ld_iter_a = smem_base_a;
  uint4* smem_ld_iter_b = smem_base_b;
  uint4* smem_st_iter_a = smem_base_a;
  uint4* smem_st_iter_b = smem_base_b;

  int pre_ld_tile       = 0;
  int pre_group         = 0;
  int cur_group         = 0;
  half* smem_st_iter_sa = smem_scale_a;
  half* smem_st_iter_sb = smem_scale_b;
  half* smem_ld_iter_sa = smem_scale_a;
  half* smem_ld_iter_sb = smem_scale_b;

  const int gmem_scale_step_a = (TileA::SYM_QUANT ? 1 : 2) * M * cu_cdiv(BK, GSIZE);
  const int gmem_scale_step_b = (TileB::SYM_QUANT ? 1 : 2) * N * cu_cdiv(BK, GSIZE);

  auto load_smem = [&]() {
    tile_a.load_smem(smem_st_iter_a, main_loop > 0);
    tile_b.load_smem(smem_st_iter_b, main_loop > 0);
    if constexpr (GSIZE != -1) {
      bool scale_pred = (pre_ld_tile % G2S_GROUP_FETCH == 0) && (main_loop > 0);
      if (scale_pred) {
        tile_a.load_scale_zp_smem(smem_st_iter_sa, scale_a, tidx, true);
        // tile_b.load_scale_zp_smem_align(smem_st_iter_sb, scale_b, tidx, true);
        tile_b.load_scale_zp_smem(smem_st_iter_sb, scale_b, tidx, true);
        // advance gmem pointer of scale
        scale_a += gmem_scale_step_a;
        scale_b += gmem_scale_step_b;
        // advance smem pointer of scale
        pre_group       = (pre_group + 1) % STAGE;
        smem_st_iter_sa = smem_scale_a + pre_group * G2S_SCALE_SIZE_A;
        smem_st_iter_sb = smem_scale_b + pre_group * G2S_SCALE_SIZE_B;
      }
    }
    // advance gmem pointer of tileAB
    tile_a.advance_gmem();
    tile_b.advance_gmem();
    // advance smem pointer of tileAB
    smem_st_stage  = (smem_st_stage + 1) % STAGE;
    smem_st_iter_a = smem_base_a + smem_st_stage * SMEM_SIZE_A;
    smem_st_iter_b = smem_base_b + smem_st_stage * SMEM_SIZE_B;

    pre_ld_tile++;
  };

  auto load_frag = [&](int k) {
    tile_a.load_frag(k, frag_a[k], smem_ld_iter_a);
    tile_b.load_frag(k, frag_b[k], smem_ld_iter_b);

    // // handle the case: GROUP_SIZE < BK; this imply: GSIZE > 1
    // if constexpr (G2S_SCALE_SIZE_A > CTA_SCALE_SIZE_A) {
    //   if (mma_iter_k % GSIZE == 0) {
    //     smem_ld_iter_sa += CTA_SCALE_SIZE_A;
    //     smem_ld_iter_sb += CTA_SCALE_SIZE_B;
    //   }
    // }

    mma_iter_k += MMA_STEP;
  };

  // 1. prologue
  if constexpr (GSIZE == -1) {
    tile_a.load_scale_zp_smem(smem_scale_a, scale_a, tidx, true);
    // tile_b.load_scale_zp_smem_align(smem_scale_b, scale_b, tidx, true);
    tile_b.load_scale_zp_smem(smem_scale_b, scale_b, tidx, true);
  }
#pragma unroll
  for (auto i = 0; i < STAGE - 1; i++) {
    load_smem();
    commit_cp_async_group();
    main_loop--;
  }
  wait_cp_async_group<STAGE - 2>();
  __syncthreads();

  load_frag(0);

  // 2. Mainloop
#pragma unroll 1
  for (; main_loop > (-STAGE + 1); main_loop--) {
    load_smem();
    commit_cp_async_group();
#pragma unroll
    for (int k = 0; k < WARP_ITER_K; ++k) {
      int pre_frag_iter = (k + 1) % WARP_ITER_K;

      if (k == WARP_ITER_K - 1) {
        // switch pre-load stage
        smem_ld_stage  = (smem_ld_stage + 1) % STAGE;
        smem_ld_iter_a = smem_base_a + smem_ld_stage * SMEM_SIZE_A;
        smem_ld_iter_b = smem_base_b + smem_ld_stage * SMEM_SIZE_B;
        tile_a.reset_frag_offset();
        tile_b.reset_frag_offset();

        if constexpr (GSIZE != -1) {
          if (mma_iter_k % GSIZE == 0) {
            cur_group       = (cur_group + 1) % STAGE;
            smem_ld_iter_sa = smem_scale_a + cur_group * G2S_SCALE_SIZE_A;
            smem_ld_iter_sb = smem_scale_b + cur_group * G2S_SCALE_SIZE_B;
            tile_c.load_scale(r_sa, r_sb, smem_ld_iter_sa, smem_ld_iter_sb, lane, warp_x, warp_y);
          }
        }

        wait_cp_async_group<STAGE - 2>();
        __syncthreads();
      }

      load_frag(pre_frag_iter);

      TileConfig::mma_compute(frag_c, frag_a[k], frag_b[k]);

      // group quant: dequant the mma integer partial sum, accumulate in float
      if constexpr (GSIZE != -1) {
        if ((mma_iter_k - MMA_STEP) % GSIZE == 0) {
          tile_c.scale_frag(r_sa, r_sb, frag_c, frag_c_out);
          tile_c.clear_frag(frag_c);
        }
      }
    }
  }
  if constexpr (GSIZE == -1) {
    tile_c.load_scale(r_sa, r_sb, smem_ld_iter_sa, smem_ld_iter_sb, lane, warp_x, warp_y);
    tile_c.scale_frag(r_sa, r_sb, frag_c, frag_c_out);
  }

  if constexpr (TileConfig::WARP_K > 1) {
    tile_c.reduce_slicek(frag_c_out, smem, lane, warp_x, warp_y, warp_z);
  }
  tile_c.template write_frag_to_gmem<T_ACC>(frag_c_out, lane, warp_x, warp_y, warp_z, smem);
}

/// special case for w4a4g128
/// reproduce atom: https://github.com/efeslab/Atom/blob/main/kernels/include/GEMM/Dense_layer_gemm_i4_o16.cuh
template <typename TileConfig, typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void cta_gemm_w4a4g128(          //
    TileA& tile_a, TileB& tile_b, TileC& tile_c,            //
    uint8_t* smem,                                          //
    const half* scale_a, const half* scale_b,               //
    half* smem_scale_a, half* smem_scale_b,                 //
    int M, int N, int K,                                    //
    int lane, int warp_x, int warp_y, int warp_z, int tidx  //
) {
  using FragA = typename TileA::FragA;
  using FragB = typename TileB::FragB;
  using FragC = typename TileC::FragC;

  using T_ACC = float;

  constexpr int BK          = TileConfig::BK;
  constexpr int STAGE       = TileConfig::STAGE;
  constexpr int WARP_K      = TileConfig::WARP_K;
  constexpr int WARP_ITER_K = TileConfig::WARP_ITER_K;

  constexpr int WARP_FRAG_M = TileConfig::WARP_FRAG_M;
  constexpr int WARP_FRAG_N = TileConfig::WARP_FRAG_N;

  constexpr int SMEM_SIZE_A = TileA::VEC_PER_ROW * TileA::TILE_ROW;
  constexpr int SMEM_SIZE_B = TileB::VEC_PER_ROW * TileB::TILE_ROW;

  // for group quant
  constexpr int GSIZE            = TileB::GSIZE;
  constexpr int G2S_SCALE_SIZE_A = TileA::G2S_SCALE_SIZE;
  constexpr int G2S_SCALE_SIZE_B = TileB::G2S_SCALE_SIZE;
  constexpr int MMA_STEP         = TileConfig::mmaK * WARP_K;

  static_assert(TileConfig::mmaK == 64, "mmaK must be 64");
  static_assert(GSIZE == BK || GSIZE == 128, "GSIZE == 128 is required");

  // allocate register frag
  FragC frag_c{};
  FragA frag_a[WARP_ITER_K];
  FragB frag_b[WARP_ITER_K];
  half r_sa[WARP_FRAG_M * 2];
  half r_sb[WARP_FRAG_N * 2];
  T_ACC frag_c_out[WARP_FRAG_M * WARP_FRAG_N][TileConfig::FRAG_C_TH]{};

  int main_loop = K / BK;

  uint4* smem_base_a = reinterpret_cast<uint4*>(smem);
  uint4* smem_base_b = reinterpret_cast<uint4*>(smem + STAGE * SMEM_SIZE_A * 16);

  int smem_ld_stage     = 0;
  int smem_st_stage     = 0;
  uint4* smem_ld_iter_a = smem_base_a;
  uint4* smem_ld_iter_b = smem_base_b;
  uint4* smem_st_iter_a = smem_base_a;
  uint4* smem_st_iter_b = smem_base_b;

  half* smem_st_iter_sa = smem_scale_a;
  half* smem_st_iter_sb = smem_scale_b;
  half* smem_ld_iter_sa = smem_scale_a;
  half* smem_ld_iter_sb = smem_scale_b;

  // int mma_iter_k = 0;

  const int gmem_scale_step_a = (TileA::SYM_QUANT ? 1 : 2) * M * cu_cdiv(BK, GSIZE);
  const int gmem_scale_step_b = (TileB::SYM_QUANT ? 1 : 2) * N * cu_cdiv(BK, GSIZE);

  auto load_smem = [&]() {
    bool scale_pred = main_loop > 0;
    tile_a.load_smem(smem_st_iter_a, main_loop > 0);
    tile_b.load_smem(smem_st_iter_b, main_loop > 0);
    tile_a.load_scale_zp_smem(smem_st_iter_sa, scale_a, tidx, scale_pred);
    // tile_b.load_scale_zp_smem_align(smem_st_iter_sb, scale_b, tidx, scale_pred);
    tile_b.load_scale_zp_smem(smem_st_iter_sb, scale_b, tidx, scale_pred);

    // advance gmem pointer of scale
    scale_a += gmem_scale_step_a;
    scale_b += gmem_scale_step_b;
    // advance smem pointer of scale
    smem_st_iter_sa = smem_scale_a + smem_st_stage * G2S_SCALE_SIZE_A;
    smem_st_iter_sb = smem_scale_b + smem_st_stage * G2S_SCALE_SIZE_B;
    // advance gmem pointer of tileAB
    tile_a.advance_gmem();
    tile_b.advance_gmem();
    // advance smem pointer of tileAB
    smem_st_stage  = (smem_st_stage + 1) % STAGE;
    smem_st_iter_a = smem_base_a + smem_st_stage * SMEM_SIZE_A;
    smem_st_iter_b = smem_base_b + smem_st_stage * SMEM_SIZE_B;
  };

  auto load_frag = [&](int k) {
    tile_a.load_frag(k, frag_a[k], smem_ld_iter_a);
    tile_b.load_frag(k, frag_b[k], smem_ld_iter_b);
    // mma_iter_k += MMA_STEP;
  };

  // 1. prologue
#pragma unroll
  for (auto i = 0; i < STAGE - 1; i++) {
    load_smem();
    commit_cp_async_group();
    main_loop--;
  }
  wait_cp_async_group<STAGE - 2>();
  __syncthreads();

  load_frag(0);

  // 2. Mainloop
#pragma unroll 1
  for (; main_loop > (-STAGE + 1); main_loop--) {
    load_frag(1);
    tile_c.load_scale(r_sa, r_sb, smem_ld_iter_sa, smem_ld_iter_sb, lane, warp_x, warp_y);
    TileConfig::mma_compute(frag_c, frag_a[0], frag_b[0]);
    load_smem();
    commit_cp_async_group();
    // switch pre-load stage
    smem_ld_stage   = (smem_ld_stage + 1) % STAGE;
    smem_ld_iter_a  = smem_base_a + smem_ld_stage * SMEM_SIZE_A;
    smem_ld_iter_b  = smem_base_b + smem_ld_stage * SMEM_SIZE_B;
    smem_ld_iter_sa = smem_scale_a + smem_ld_stage * G2S_SCALE_SIZE_A;
    smem_ld_iter_sb = smem_scale_b + smem_ld_stage * G2S_SCALE_SIZE_B;
    tile_a.reset_frag_offset();
    tile_b.reset_frag_offset();
    wait_cp_async_group<STAGE - 2>();
    __syncthreads();
    TileConfig::mma_compute(frag_c, frag_a[1], frag_b[1]);
    load_frag(0);
    tile_c.scale_frag(r_sa, r_sb, frag_c, frag_c_out);
    tile_c.clear_frag(frag_c);

    // #pragma unroll
    //     for (int k = 0; k < WARP_ITER_K; k++) {
    //       int pre_frag_iter = (k + 1) % WARP_ITER_K;
    //       if (k == WARP_ITER_K - 1) {
    //         // switch pre-load stage
    //         smem_ld_stage  = (smem_ld_stage + 1) % STAGE;
    //         smem_ld_iter_a = smem_base_a + smem_ld_stage * SMEM_SIZE_A;
    //         smem_ld_iter_b = smem_base_b + smem_ld_stage * SMEM_SIZE_B;
    //         cur_group       = (cur_group + 1) % STAGE;
    //         smem_ld_iter_sa = smem_scale_a + cur_group * G2S_SCALE_SIZE_A;
    //         smem_ld_iter_sb = smem_scale_b + cur_group * G2S_SCALE_SIZE_B;
    //         tile_a.reset_frag_offset();
    //         tile_b.reset_frag_offset();
    //         wait_cp_async_group<STAGE - 2>();
    //         __syncthreads();
    //       }
    //       load_frag(pre_frag_iter);
    //       if (k == 0) {
    //         load_smem();
    //         commit_cp_async_group();
    //         tile_c.load_scale(r_sa, r_sb, smem_ld_iter_sa, smem_ld_iter_sb, lane, warp_x, warp_y);
    //       }
    //       TileConfig ::mma_compute(frag_c, frag_a[k], frag_b[k]);
    //       if (k == 1) {
    //         tile_c.scale_frag(r_sa, r_sb, frag_c, frag_c_out);
    //         tile_c.clear_frag(frag_c);
    //       }
    //     }
  }

  tile_c.template write_frag_to_gmem<T_ACC>(frag_c_out, lane, warp_x, warp_y, warp_z, smem);
}

}  // namespace tiled_gemm