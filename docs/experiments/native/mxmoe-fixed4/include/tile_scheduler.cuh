#pragma once

#include "cuda_utils.cuh"

struct TileScheduler {
  int problem_count{};
  int num_group_tiles{};
  int* problem_tiles_prefix_sum{};
  int* problem_bitwidth{};
  dim3* problem_sizes{};

  __host__ __device__ TileScheduler(  //
      dim3* _problem_sizes,           //
      int _problem_count,             //
      int* _problem_tiles_prefix_sum  //
  ) {
    problem_tiles_prefix_sum = _problem_tiles_prefix_sum;
    num_group_tiles          = _problem_tiles_prefix_sum[_problem_count - 1];
    problem_count            = _problem_count;
    problem_sizes            = _problem_sizes;
  }

  /// if tile_idx > total_tiles, return -1
  /// else return the problem index
  __host__ __device__ auto get_problem_idx(int tile_idx) {
    auto problem_idx = -1;
    if (tile_idx >= num_group_tiles) return problem_idx;

    for (auto i = 0; i < problem_count; i++) {
      if (tile_idx >= problem_tiles_prefix_sum[i]) {
        continue;
      }
      problem_idx = i;
      break;
    }
    return problem_idx;
  }

  template <int BM, int BN>
  __host__ __device__ auto get_tile_coord(int problem_idx, int tile_idx) {
    auto problem_size = problem_sizes[problem_idx];
    int M             = problem_size.x;
    int N             = problem_size.y;

    auto problem_grid_shape = dim3(cu_cdiv(M, BM), cu_cdiv(N, BN), 1);

    if (problem_idx != 0) tile_idx -= problem_tiles_prefix_sum[problem_idx - 1];

    return dim3(tile_idx / problem_grid_shape.y, tile_idx % problem_grid_shape.y, 0);
  }
};