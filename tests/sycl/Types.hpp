#ifndef TYPES_HPP
#define TYPES_HPP

#include <array>
#include <vector>
#include <complex>
#include <experimental/mdspan>
#include <sycl/View.hpp>

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
using complex64 = std::complex<float32>;
using complex128 = std::complex<float64>;

using Real = float64;
using size_type = std::size_t;

template <size_type N>
using shape_type = std::array<size_type, N>;

namespace stdex = std::experimental;

#if defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__)
  using default_layout = stdex::layout_left;
#else
  using default_layout = stdex::layout_left;
#endif

template <typename ElementType>
using Mdspan1D = stdex::mdspan<ElementType, std::dextents<size_type, 1>, default_layout>;
template <typename ElementType>
using Mdspan2D = stdex::mdspan<ElementType, std::dextents<size_type, 2>, default_layout>;
template <typename ElementType>
using Mdspan3D = stdex::mdspan<ElementType, std::dextents<size_type, 3>, default_layout>;
template <typename ElementType>
using Mdspan4D = stdex::mdspan<ElementType, std::dextents<size_type, 4>, default_layout>;

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

using ComplexView1D = View1D< std::complex<Real> >;
using ComplexView2D = View2D< std::complex<Real> >;
using ComplexView3D = View3D< std::complex<Real> >;

using IntView1D = View1D<int>;
using IntView2D = View2D<int>;

#endif