#ifndef __EXECUTORS_TRANSPOSE_HPP__
#define __EXECUTORS_TRANSPOSE_HPP__

#include <cmath>
#include <cassert>
#include <array>
#include <vector>
#include <experimental/mdspan>
#include "../Iteration.hpp"
#include "Parallel_For.hpp"
#if defined(ENABLE_OPENMP)
  #include "../openmp_linalg.hpp"
#elif defined(_NVHPC_CUDA) || defined(__CUDACC__)
  #include "../cuda_linalg.hpp"
#else
  #include "../openmp_linalg.hpp"
#endif

namespace Impl {
  /* Transpose batched matrix */
  template <class InputView, class OutputView,
          std::enable_if_t<InputView::rank()==3 && OutputView::rank()==3, std::nullptr_t> = nullptr>
  void transpose(const blasHandle_t& blas_handle, const InputView& in, OutputView& out, const std::array<int, 3>& axes) {
    static_assert( std::is_same_v<typename InputView::value_type, typename OutputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename OutputView::layout_type> );
    using value_type = InputView::value_type;
    using size_type = InputView::size_type;
    using layout_type = InputView::layout_type;
    using axes_type = std::array<int, 3>;

    for(std::size_t i=0; i<axes.size(); i++) {
      assert(out.extent(i) == in.extent(axes[i]));
    }

    const auto n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2);
    // Not quite sure, this is a better strategy
    IteratePolicy<typename InputView::layout_type, 3> policy3d({0, 0, 0}, {n0, n1, n2});

    if(axes == axes_type({0, 1, 2}) ) {
      const auto n = in.size();
      thrust::copy(thrust::device, in.data_handle(), in.data_handle()+n, out.data_handle());
    } else if(axes == axes_type({0, 2, 1}) ) {
      Impl::for_each(policy3d,
        [=](const int i0, const int i1, const int i2) {
          out(i0, i2, i1) = in(i0, i1, i2);
      });
    } else if(axes == axes_type({1, 0, 2})) {
      Impl::for_each(policy3d,
        [=](const int i0, const int i1, const int i2) {
          out(i1, i0, i2) = in(i0, i1, i2);
      });
    } else if(axes == axes_type({1, 2, 0})) {
      using mdspan2d_type = stdex::mdspan<value_type, stdex::dextents<size_type, 2>, layout_type>;
      using extent2d_type = std::array<std::size_t, 2>;
      extent2d_type in_shape({in.extent(0), in.extent(1) * in.extent(2)});
      extent2d_type out_shape({in.extent(1) * in.extent(2), in.extent(0)});

      const mdspan2d_type sub_in(in.data_handle(), in_shape);
      mdspan2d_type sub_out(out.data_handle(), out_shape);
      transpose(blas_handle, sub_in, sub_out);
    } else if(axes == axes_type({2, 0, 1})) {
      using mdspan2d_type = stdex::mdspan<value_type, stdex::dextents<size_type, 2>, layout_type>;
      using extent2d_type = std::array<std::size_t, 2>;
      extent2d_type in_shape({in.extent(0) * in.extent(1), in.extent(2)});
      extent2d_type out_shape({in.extent(2), in.extent(0) * in.extent(1)});

      const mdspan2d_type sub_in(in.data_handle(), in_shape);
      mdspan2d_type sub_out(out.data_handle(), out_shape);
      transpose(blas_handle, sub_in, sub_out);
    } else if(axes == axes_type({2, 1, 0})) {
      Impl::for_each(policy3d,
        [=](const int i0, const int i1, const int i2) {
          out(i2, i1, i0) = in(i0, i1, i2);
      });
    } else {
      std::runtime_error("Invalid axes specified.");
    }
  }

  /* Transpose batched matrix */
  template <class InputView, class OutputView,
          std::enable_if_t<InputView::rank()==3 && OutputView::rank()==3, std::nullptr_t> = nullptr>
  void transpose(const InputView& in, OutputView& out, const std::array<int, 3>& axes) {
    static_assert( std::is_same_v<typename InputView::value_type, typename OutputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename OutputView::layout_type> );
    Impl::blasHandle_t blas_handle;
    blas_handle.create();
    transpose(blas_handle, in, out, axes);
    blas_handle.destroy();
  }
};

#endif
