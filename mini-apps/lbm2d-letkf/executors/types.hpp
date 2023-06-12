#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <array>
#include <experimental/mdspan>
#include <Iteration.hpp>
#include <executors/View.hpp>

namespace stdex = std::experimental;
  
#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
  #include <thrust/complex.h>
  #define SIMD_LOOP
  #define SIMD_WIDTH 1
  using default_layout = stdex::layout_left;
  using default_iterate_layout = stdex::layout_left;
  template <typename RealType> using Complex = thrust::complex<RealType>;
#else
  #include <complex>
  using default_layout = stdex::layout_right;
  using default_iterate_layout = stdex::layout_right;
  template <typename RealType> using Complex = std::complex<RealType>;
  #define SIMD_WIDTH 8
  #include<omp.h>
  #if defined(SIMD)
    #define SIMD_LOOP _Pragma("omp simd")
  #else
    #define SIMD_LOOP
  #endif
#endif

using float32 = float;
using float64 = double;
using complex64 = Complex<float32>;
using complex128 = Complex<float64>;
using size_type = std::size_t;

using Real = float64;

template <size_type N>
using shape_type = std::array<size_type, N>;

template < typename ElementType >
using View1D = View<ElementType, stdex::dextents<size_type, 1>, default_layout>;

template < typename ElementType >
using View2D = View<ElementType, stdex::dextents<size_type, 2>, default_layout>;

template < typename ElementType >
using View3D = View<ElementType, stdex::dextents<size_type, 3>, default_layout>;

using RealView1D = View1D<Real>;
using RealView2D = View2D<Real>;
using RealView3D = View3D<Real>;

template <size_t ND>
using Iterate_policy = IteratePolicy<default_iterate_layout, ND>;

#endif
