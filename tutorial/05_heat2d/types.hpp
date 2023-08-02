#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <thrust/execution_policy.h>
#include <experimental/mdspan>

using counting_iterator = thrust::counting_iterator<std::size_t>;
namespace stdex = std::experimental;

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
  using default_layout = stdex::layout_left;
#else
  using default_layout = stdex::layout_left;
#endif

using RealView1D = stdex::mdspan<double, stdex::dextents<std::size_t, 1>, default_layout>;
using RealView2D = stdex::mdspan<double, stdex::dextents<std::size_t, 2>, default_layout>;
using RealView3D = stdex::mdspan<double, stdex::dextents<std::size_t, 3>, default_layout>;

#endif
