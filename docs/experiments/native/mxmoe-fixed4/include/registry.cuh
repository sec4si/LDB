#pragma once
#include <unordered_map>

#include "cuda_utils.cuh"
#include "fmt/ranges.h"
#include "mm_tile.cuh"
#include "quantize.cuh"

namespace mxmoe {
template <class T>
constexpr std::string_view type_name() {
  using namespace std;
#ifdef __clang__
  string_view p = __PRETTY_FUNCTION__;
  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
  string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#else
  return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
  string_view p = __FUNCSIG__;
  return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

using FuncType = void (*)(                                       //
    half** ptr_As, half** ptr_Bs,                                //
    half** ptr_scale_a,                                          //
    half** ptr_scale_b,                                          //
    half** ptr_Cs, half** ptr_Ds,                                //
    int64_t* ldas, int64_t* ldbs, int64_t* ldcs, int64_t* ldds,  //
    dim3* device_problem_sizes,                                  //
    dim3* host_problem_sizes,                                    //
    QParams* device_qbits_list,                                  //
    QParams* host_qbits_list,                                    //
    int problem_count                                            //
);

template <typename Tile>
auto stringfy_tile_cfg() {
  return fmt::format("({},{},{}) ({},{},{}) {}", Tile::BM, Tile::BN, Tile::BK, Tile::WARP_M, Tile::WARP_N, Tile::WARP_K,
                     Tile::STAGE);
}

template <typename Tile>
auto stringfy_qcfg() {
  using QA = typename Tile::QConfigA;
  using QB = typename Tile::QConfigB;

  auto res = std::string("w{w_bits}a{}_g{}_{}");

  if constexpr (std::is_same_v<tiled_gemm::NO_QUANT, QA> && std::is_same_v<tiled_gemm::NO_QUANT, QB>) {
    return fmt::format("w16a16");
  } else {
    if constexpr (std::is_same_v<tiled_gemm::NO_QUANT, QA>) {
      return fmt::format("w{}a16_g{}_{}", QB::QBITS, QB::GSIZE, QB::SYM ? "sym" : "asym");
    } else {
      static_assert(QB::GSIZE == QA::GSIZE, "GSIZE must be the same for A and B");
      static_assert(QB::SYM == QA::SYM, "SYM must be the same for A and B");
      return fmt::format("w{}a{}_g{}_{}", QB::QBITS, QA::QBITS, QA::GSIZE, QA::SYM ? "sym" : "asym");
    }
  }
}

template <typename... Tiles>
auto stringfy_tile_cfgs() {
  return fmt::format("{}", (fmt::format("[{}: {}], ", stringfy_qcfg<Tiles>(), stringfy_tile_cfg<Tiles>()) + ...));
}

struct KernelRegistry {
  struct Kernel {
    std::string cfg_str;
    FuncType func;
    Kernel(std::string cfg_str, FuncType func) : cfg_str(cfg_str), func(func) {}
  };

  std::vector<Kernel> storage_;
  std::unordered_map<std::string, FuncType> kernel_map_;

  template <typename... TileConfigs>
  void register_kernel(FuncType func, std::string fname) {
    auto kernel = Kernel(stringfy_tile_cfgs<TileConfigs...>() + fname, func);
    storage_.push_back(kernel);
    kernel_map_[kernel.cfg_str] = func;
  }

  const auto& operator[](size_t i) { return storage_[i]; }

  void print() {
    for (auto i = 0; i < storage_.size(); i++) {
      fmt::println("{}: {}", i, storage_[i].cfg_str);
    }
    fmt::println("");
  }
};

inline KernelRegistry& GetGlobalRegistry() {
  static KernelRegistry instance;
  return instance;
}

template <typename... Tiles, typename F>
inline auto register_kernel(F f, std::string name) {
  GetGlobalRegistry().register_kernel<Tiles...>(f, name);
}

}  // namespace mxmoe
