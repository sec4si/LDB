#pragma once

#include <cuda_fp16.h>

namespace tiled_gemm {
enum class Layout { RowMajor, ColMajor };

template <                                //
    Layout LAYOUT_A_ = Layout::RowMajor,  //
    Layout LAYOUT_B_ = Layout::RowMajor,  //
    Layout LAYOUT_C_ = Layout::RowMajor,  //
    typename TA_     = half,              //
    typename TB_     = half,              //
    typename TC_     = float              //
    >
struct MatricesInfo {
  using TA                         = TA_;
  static constexpr Layout LAYOUT_A = LAYOUT_A_;
  using TB                         = TB_;
  static constexpr Layout LAYOUT_B = LAYOUT_B_;
  using TC                         = TC_;
  static constexpr Layout LAYOUT_C = LAYOUT_C_;
};

// A RowMajor, B RowMajor, C RowMajor
using RRR_FP16FP16FP32 = MatricesInfo<Layout::RowMajor, Layout::RowMajor, Layout::RowMajor, half, half, float>;
using RRR_FP16FP16FP16 = MatricesInfo<Layout::RowMajor, Layout::RowMajor, Layout::RowMajor, half, half, half>;
// A RowMajor, B Colmajor, C RowMajor
using RCR_FP16FP16FP32 = MatricesInfo<Layout::RowMajor, Layout::ColMajor, Layout::RowMajor, half, half, float>;
using RCR_FP16FP16FP16 = MatricesInfo<Layout::RowMajor, Layout::ColMajor, Layout::RowMajor, half, half, half>;
using RCR_U16U16FP16     = MatricesInfo<Layout::RowMajor, Layout::ColMajor, Layout::RowMajor, uint16_t, uint16_t, half>;
}  // namespace tiled_gemm