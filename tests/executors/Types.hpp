#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <array>
#include <experimental/mdspan>
#include <Iteration.hpp>
#include <executors/View.hpp>

namespace stdex = std::experimental;

#if defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__)
  #include <thrust/complex.h>
  #define SIMD_LOOP
  #define SIMD_WIDTH 1
  using default_layout = stdex::layout_left;
  using default_iterate_layout = stdex::layout_left;
  template <typename RealType> using Complex = thrust::complex<RealType>;
#else
  #include <complex>
  using default_layout = stdex::layout_left;
  using default_iterate_layout = stdex::layout_left;
  template <typename RealType> using Complex = std::complex<RealType>;
  #define SIMD_WIDTH 8
  #include<omp.h>
  #if defined(SIMD)
    #define SIMD_LOOP _Pragma("omp simd")
  #else
    #define SIMD_LOOP
  #endif
#endif

using int8  = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using uint8  = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

using float32 = float;
using float64 = double;
using complex64 = Complex<float32>;
using complex128 = Complex<float64>;

using Real = float64;
using size_type = uint32;

template <size_type N>
using shape_type = std::array<size_type, N>;

template < typename ElementType > 
using View1D = View<ElementType, std::dextents< size_type, 1 >, default_layout >;
template < typename ElementType > 
using View2D = View<ElementType, std::dextents< size_type, 2 >, default_layout >;
template < typename ElementType > 
using View3D = View<ElementType, std::dextents< size_type, 3 >, default_layout >;
template < typename ElementType > 
using View4D = View<ElementType, std::dextents< size_type, 4 >, default_layout >;

using RealView1D = View1D<Real>;
using RealView2D = View2D<Real>;
using RealView3D = View3D<Real>;
using RealView4D = View4D<Real>;

using ComplexView1D = View1D< Complex<Real> >;
using ComplexView2D = View2D< Complex<Real> >; 
using ComplexView3D = View3D< Complex<Real> >;

template <size_t ND>
using Iterate_policy = IteratePolicy<default_iterate_layout, ND>;

#endif
