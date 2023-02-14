#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <array>
#include <experimental/mdspan>

using size_type = std::size_t;

template <size_type N>
using shape_type = std::array<size_type, N>;

namespace stdex = std::experimental;

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
  using default_layout = stdex::layout_left;
#else
  using default_layout = stdex::layout_right;
#endif

template <typename ElementType>
using View1D = stdex::mdspan<ElementType, stdex::dextents<size_type, 1>, default_layout>;

template <typename ElementType>
using View2D = stdex::mdspan<ElementType, stdex::dextents<size_type, 2>, default_layout>;

template <typename ElementType>
using View3D = stdex::mdspan<ElementType, stdex::dextents<size_type, 3>, default_layout>;

#endif
