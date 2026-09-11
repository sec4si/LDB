#pragma once
#include <cuda_fp16.h>
#include <mma.h>

#include <tuple>

namespace nv_precision = nvcuda::wmma::experimental::precision;

static inline constexpr int WarpSize = 32;

template <typename T>
constexpr __host__ __device__ __forceinline__ T cu_cdiv(T a, T b) {
  static_assert(std::is_integral_v<T>, "T must be integral type");
  return (a + b - 1) / b;
}

template <typename T>
__host__ __device__ __forceinline__ constexpr int bitsof() {
  return sizeof(T) * 8;
}

template <>
__host__ __device__ __forceinline__ constexpr int bitsof<nv_precision::s4>() {
  return 4;
}
template <>
__host__ __device__ __forceinline__ constexpr int bitsof<nv_precision::u4>() {
  return 4;
}
template <>
__host__ __device__ __forceinline__ constexpr int bitsof<nv_precision::b1>() {
  return 1;
}

#ifndef OFFSET
// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#endif

__device__ __forceinline__ auto cvt_to_smem_addr(const void* p) { return __cvta_generic_to_shared(p); }

template <int N_BYTES = 16>
__device__ __forceinline__ void cp_async_ca(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(cvt_to_smem_addr(dst)), "l"(src),
               "n"(N_BYTES));
}

template <int N_BYTES = 16>
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(cvt_to_smem_addr(dst)), "l"(src),
               "n"(N_BYTES));
}

template <int N_BYTES = 16>
__device__ inline void cp_async_stream(const void* dst, const void* src) {
  asm volatile(
      "{\n"
      "   .reg .b64 p;\n"
      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
      "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
      "}\n" ::"l"(cvt_to_smem_addr(dst)),
      "l"(src), "n"(N_BYTES));
}

__device__ __forceinline__ void commit_cp_async_group() { asm("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wait_cp_async_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int L2_PREFETCH = 128, int N_BYTES = 16>
__device__ __forceinline__ void cp_async_cg_pred(const void* dst, const void* src, bool pred) {
  if constexpr (L2_PREFETCH == 256) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %0, 0;\n"
        // "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
        "@p cp.async.cg.shared.global.L2::256B [%1], [%2], %3;\n"
        "}\n" ::"r"((int)pred),
        "l"(cvt_to_smem_addr(dst)), "l"(src), "n"(N_BYTES));
  } else if constexpr (L2_PREFETCH == 128) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %0, 0;\n"
        // "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
        "@p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
        "}\n" ::"r"((int)pred),
        "l"(cvt_to_smem_addr(dst)), "l"(src), "n"(N_BYTES));
  } else if constexpr (L2_PREFETCH == 64) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %0, 0;\n"
        // "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
        "@p cp.async.cg.shared.global.L2::64B [%1], [%2], %3;\n"
        "}\n" ::"r"((int)pred),
        "l"(cvt_to_smem_addr(dst)), "l"(src), "n"(N_BYTES));
  } else {
    static_assert(L2_PREFETCH == 256 || L2_PREFETCH == 128 || L2_PREFETCH == 64, "L2_PREFETCH: [256 or 128 or 64]");
  }
}

__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t R[4], const void* smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"((uint32_t)cvt_to_smem_addr(smem_ptr)));
}

__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t R[4], const void* smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"((uint32_t)cvt_to_smem_addr(smem_ptr)));
}

__device__ __forceinline__ void ldmatrix_m8n8x2(uint32_t R[2], const void* smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"((uint32_t)cvt_to_smem_addr(smem_ptr)));
}

__device__ __forceinline__ void ldmatrix_m8n8x2_trans(uint32_t R[2], const void* smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"((uint32_t)cvt_to_smem_addr(smem_ptr)));
}

__device__ __forceinline__ void ldmatrix_m8n8(uint32_t R[1], const void* smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(R[0])
               : "r"((uint32_t)cvt_to_smem_addr(smem_ptr)));
}

__device__ __forceinline__ void ldmatrix_m8n8_trans(uint32_t R[1], const void* smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(R[0])
               : "r"((uint32_t)cvt_to_smem_addr(smem_ptr)));
}

template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n16k16_fp16fp32(  //
    uint32_t& c0, uint32_t& c1,                          //
    uint32_t& c2, uint32_t& c3,                          //
    uint32_t& c4, uint32_t& c5,                          //
    uint32_t& c6, uint32_t& c7,                          //
    const uint32_t& a0, const uint32_t& a1,              //
    const uint32_t& a2, const uint32_t& a3,              //
    const uint32_t& b0, const uint32_t& b1,              //
    const uint32_t& b2, const uint32_t& b3               //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b2), "r"(b3),                       //
          "r"(c4), "r"(c5), "r"(c6), "r"(c7));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b2), "r"(b3),                       //
          "r"(0), "r"(0), "r"(0), "r"(0));
  }
}

template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n16k16_fp16fp16(  //
    uint32_t& c0, uint32_t& c1,                          //
    uint32_t& c2, uint32_t& c3,                          //
    const uint32_t& a0, const uint32_t& a1,              //
    const uint32_t& a2, const uint32_t& a3,              //
    const uint32_t& b0, const uint32_t& b1,              //
    const uint32_t& b2, const uint32_t& b3               //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c0), "=r"(c1)                   //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),  //
          "r"(b0), "r"(b1),                    //
          "r"(c0), "r"(c1));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c2), "=r"(c3)                   //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),  //
          "r"(b2), "r"(b3),                    //
          "r"(c2), "r"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c0), "=r"(c1)                   //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),  //
          "r"(b0), "r"(b1),                    //
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c2), "=r"(c3)                   //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),  //
          "r"(b2), "r"(b3),                    //
          "r"(0), "r"(0));
  }
}

template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n8k16_fp16fp32(  //
    uint32_t& c0, uint32_t& c1,                         //
    uint32_t& c2, uint32_t& c3,                         //
    const uint32_t& a0, const uint32_t& a1,             //
    const uint32_t& a2, const uint32_t& a3,             //
    const uint32_t& b0, const uint32_t& b1              //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(0), "r"(0), "r"(0), "r"(0));
  }
}

template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n8k16_bf16fp32(  //
    uint32_t& c0, uint32_t& c1,                         //
    uint32_t& c2, uint32_t& c3,                         //
    const uint32_t& a0, const uint32_t& a1,             //
    const uint32_t& a2, const uint32_t& a3,             //
    const uint32_t& b0, const uint32_t& b1              //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(0), "r"(0), "r"(0), "r"(0));
  }
}

template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n8k16_fp16fp16(  //
    uint32_t& c0, uint32_t& c1,                         //
    const uint32_t& a0, const uint32_t& a1,             //
    const uint32_t& a2, const uint32_t& a3,             //
    const uint32_t& b0, const uint32_t& b1              //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c0), "=r"(c1)                   //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),  //
          "r"(b0), "r"(b1),                    //
          "r"(c0), "r"(c1));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(c0), "=r"(c1)                   //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),  //
          "r"(b0), "r"(b1),                    //
          "r"(0), "r"(0));
  }
}

template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n8k32_s8s32(  //
    uint32_t& c0, uint32_t& c1,                      //
    uint32_t& c2, uint32_t& c3,                      //
    const uint32_t& a0, const uint32_t& a1,          //
    const uint32_t& a2, const uint32_t& a3,          //
    const uint32_t& b0, const uint32_t& b1           //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(0), "r"(0), "r"(0), "r"(0));
  }
}

#if COMPUTE_ARCH == 89
template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n8k32_e4m3fp32(  //
    uint32_t& c0, uint32_t& c1,                         //
    uint32_t& c2, uint32_t& c3,                         //
    const uint32_t& a0, const uint32_t& a1,             //
    const uint32_t& a2, const uint32_t& a3,             //
    const uint32_t& b0, const uint32_t& b1              //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(0), "r"(0), "r"(0), "r"(0));
  }
}
#endif

template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n8k64_s4s32(  //
    uint32_t& c0, uint32_t& c1,                      //
    uint32_t& c2, uint32_t& c3,                      //
    const uint32_t& a0, const uint32_t& a1,          //
    const uint32_t& a2, const uint32_t& a3,          //
    const uint32_t& b0, const uint32_t& b1           //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(0), "r"(0), "r"(0), "r"(0));
  }
}

template <bool INIT_ZERO = false>
__device__ __forceinline__ void mma_m16n8k256_b1s32(  //
    uint32_t& c0, uint32_t& c1,                      //
    uint32_t& c2, uint32_t& c3,                      //
    const uint32_t& a0, const uint32_t& a1,          //
    const uint32_t& a2, const uint32_t& a3,          //
    const uint32_t& b0, const uint32_t& b1           //
) {
  if constexpr (!INIT_ZERO) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.b1.b1.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.b1.b1.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)  //
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     //
          "r"(b0), "r"(b1),                       //
          "r"(0), "r"(0), "r"(0), "r"(0));
  }
}


template <int LD, int BANK_WIDTH = 64>
__device__ __forceinline__ auto _smem_swizzle_map(int logic_row, int logic_col) {
  if constexpr (LD == 16) {
    auto physic_row = logic_row / 4;                     // LD=16
    auto physic_col = logic_col + (logic_row % 4) * LD;  // LD=16
    physic_col ^= (physic_row % 2) << 3;
    return physic_col + BANK_WIDTH * physic_row;
  } else if constexpr (LD == 32) {
    auto physic_row = logic_row / 2;                     // LD=32
    auto physic_col = logic_col + (logic_row % 2) * LD;  // LD=32
    physic_col ^= ((physic_row % 4) << 3);
    return physic_col + BANK_WIDTH * physic_row;
  } else if constexpr (LD == 64) {
    auto physic_row = logic_row;  // LD=64
    auto physic_col = logic_col;  // LD=64
    physic_col ^= ((physic_row % 8) << 3);
    return physic_col + BANK_WIDTH * physic_row;
  } else if constexpr (LD == 128) {
    auto physic_row = logic_row * 2 + logic_col / 64;  // LD=128
    auto physic_col = logic_col % 64;                  // LD=128
    physic_col ^= (physic_row / 2 % 8) << 3;
    return physic_col + BANK_WIDTH * physic_row;
  } else if constexpr (LD == 256) {
    auto physic_row = logic_row * 4 + logic_col / 64;  // LD=256
    auto physic_col = logic_row % 64;                  // LD=256
    physic_col ^= (physic_row / 4 % 8) << 3;
    return physic_col + BANK_WIDTH * physic_row;
  } else {
    return -1;
  }
}

/// @deprecated
/// DIV_FACTOR: number of rows required in shared memory for every `8x8` tile (fp16)
template <int TILE_LD, typename DTYPE = half>
__host__ __device__ __forceinline__ int smem_swizzle_map(int target_tile_row, int target_tile_col) {
  static constexpr int SMEM_BANK_WIDTH = 128;

  // ldmatrix requires a 16-byte continuous memory address for one thread
  constexpr int GRANULARITY   = 16 / sizeof(DTYPE);
  constexpr int TILE_LD_BYTES = TILE_LD * sizeof(DTYPE);
  constexpr int SMEM_LD       = SMEM_BANK_WIDTH / sizeof(DTYPE);

  // swizzle constrain: TILE_LD_BYTES >= 16 && TILE_LD_BYTES % 16 == 0
  static_assert(TILE_LD_BYTES % 16 == 0, "");

  constexpr int DIV_FACTOR = [] {
    // 1. each row in smem has `DIV_FACTOR` rows in logic tile
    if constexpr (TILE_LD_BYTES <= SMEM_BANK_WIDTH) {
      static_assert(SMEM_BANK_WIDTH % TILE_LD_BYTES == 0, "");
      return SMEM_BANK_WIDTH / TILE_LD_BYTES;
      // 2. each row in logic tile occupies `DIV_FACTOR` row in smem
    } else {
      static_assert(TILE_LD_BYTES % SMEM_BANK_WIDTH == 0, "");
      return TILE_LD_BYTES / SMEM_BANK_WIDTH;
    }
  }();

  int tile_off   = OFFSET(target_tile_row, target_tile_col, TILE_LD);
  int physic_row = tile_off / SMEM_LD;

  int xor_factor = [physic_row] {
    if constexpr (TILE_LD_BYTES <= SMEM_BANK_WIDTH) {
      return physic_row % (8 / DIV_FACTOR);
    } else {
      return physic_row / DIV_FACTOR % GRANULARITY;
    }
  }();
  int physic_col = ((tile_off % SMEM_LD / GRANULARITY) ^ xor_factor) * GRANULARITY;

  return OFFSET(physic_row, physic_col, SMEM_LD);
}

/// @deprecated
template <int TILE_LD>
__host__ __device__ __forceinline__ int smem_sequential_map(int target_tile_row, int target_tile_col) {
  int tile_off   = OFFSET(target_tile_row, target_tile_col, TILE_LD);
  int physic_row = tile_off / (8 * 8);
  int physic_col = tile_off % (8 * 8);
  int physic_off = OFFSET(physic_row, physic_col, 8 * 8);
  return physic_off;
}

// clang-format off
template <int... Strides, typename... Indices>
__host__ __device__ auto OFF(Indices... indices) {
  static_assert(
    (sizeof...(Strides) == sizeof...(Indices)
      && std::get<sizeof...(Strides) - 1>(std::make_tuple(Strides...)) == 1)
  ||(sizeof...(Strides) == sizeof...(Indices) - 1),"");

  if constexpr (sizeof...(Strides)==sizeof...(indices))
    return ((Strides * indices) + ...);
  else
    return OFF<Strides..., 1>(indices...);
}
// clang-format on

// transfer 16 bytes
#define FETCH_FLOAT4(pointer) (const_cast<float4*>(reinterpret_cast<const float4*>(&(pointer)))[0])
#define FETCH_FLOAT2(pointer) (const_cast<float2*>(reinterpret_cast<const float2*>(&(pointer)))[0])
#define FETCH_FLOAT(pointer) (const_cast<float*>(reinterpret_cast<const float*>(&(pointer)))[0])

template <int VEC_SIZE, typename T>
__device__ __forceinline__ void vec_move(T* dst, const T* src) {
  if constexpr (sizeof(float4) / sizeof(T) == VEC_SIZE) {
    FETCH_FLOAT4(dst[0]) = FETCH_FLOAT4(src[0]);
  } else if constexpr (sizeof(float2) / sizeof(T) == VEC_SIZE) {
    FETCH_FLOAT2(dst[0]) = FETCH_FLOAT2(src[0]);
  } else if constexpr (sizeof(float) / sizeof(T) == VEC_SIZE) {
    FETCH_FLOAT(dst[0]) = FETCH_FLOAT(src[0]);
  } else {
    dst[0] = src[0];
  }
}

template <typename T>
__device__ __forceinline__ T cu_add(T a, T b) {
  return a + b;
}

template <typename T>
__device__ __forceinline__ T cu_max(T a, T b) {
  if constexpr (std::is_same_v<T, half>)
    return __hmax(a, b);
  else if constexpr (std::is_same_v<T, float>)
    return fmaxf(a, b);
  else if constexpr (std::is_same_v<T, double>)
    return fmax(a, b);
  else
    static_assert(!std::is_same_v<T, T>, "not implemented");
}

template <typename T>
__device__ __forceinline__ T cu_min(T a, T b) {
  if constexpr (std::is_same_v<T, half>)
    return __hmin(a, b);
  else if constexpr (std::is_same_v<T, float>)
    return fminf(a, b);
  else if constexpr (std::is_same_v<T, double>)
    return fmin(a, b);
  else
    static_assert(!std::is_same_v<T, T>, "not implemented");
}

template <typename T, T (*Reduce)(T, T), int WarpSize = 32>
__device__ __forceinline__ T warp_reduce(T val) {
  // less than 32 and is power of 2
  static_assert(WarpSize <= 32 && ((WarpSize & (WarpSize - 1)) == 0), "warpsize must <= 32 and is power of 2");
#pragma unroll
  for (int i = WarpSize / 2; i >= 1; i /= 2) {
    val = Reduce(__shfl_xor_sync(0xffffffff, val, i, WarpSize), val);
  }
  return val;
}

using b128_t = uint4;

enum class SwizzleMode {
  k64B,
  k128B,
};

/// from flashinfer: https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/permuted_smem.cuh
template <SwizzleMode swizzle_mode>
struct smem_t {
  // // The base pointer.
  // b128_t* base;
  // __device__ __forceinline__ smem_t() : base(nullptr) {}
  // template <typename T>
  // __device__ __forceinline__ smem_t(T* base) : base((b128_t*)base) {}

  /*!
   * \brief Compute the element offset given coordinates in a permuted shared memory.
   * \tparam stride The stride (in terms of b128_t's) in the permuted shared memory.
   * \param i The row index.
   * \param j The column index.
   */
  template <uint32_t stride>
  static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i, uint32_t j) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      return i * stride + (j ^ (i % 8));
    } else {
      // swizzle_mode == SwizzleMode::k64B
      static_assert(stride == 4);
      return i * stride + (j ^ ((i / 2) % 4));
    }
  }

  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(uint32_t offset, uint32_t step_idx) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + (step_idx % 4 == 3) * 8;
      } else if constexpr (step_size == 4) {
        return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
      } else {
        // step_size % 8 == 0
        return offset + step_size;
      }
    } else {
      // swizzle_mode == SwizzleMode::k64B
      static_assert(step_size == 2, "Unsupported step size");
      return (offset ^ 0x2) + (step_idx % 2 == 1) * 4;
    }
  }

  template <uint32_t step_size, uint32_t row_stride>
  static __device__ __forceinline__ uint32_t advance_offset_by_row(uint32_t offset) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x4) + step_size * row_stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * row_stride;
      }
    } else {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x2) + step_size * row_stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * row_stride;
      }
    }
  }
};