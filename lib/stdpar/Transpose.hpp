#ifndef __STDPAR_TRANSPOSE_HPP__
#define __STDPAR_TRANSPOSE_HPP__

#include <cmath>
#include <cassert>
#include <array>
#include <vector>
#include <experimental/mdspan>
#include <cublas_v2.h>
#include "../Iteration.hpp"
#include "Parallel_For.hpp"

namespace Impl {
  template <typename T>
  cublasStatus_t geam(cublasHandle_t handle,
          cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n,
          const T* alpha,
          const T* A, int lda,
          const T* beta,
          const T* B, int ldb,
          T* C, int ldc
      );
  
  template <>
  cublasStatus_t geam(cublasHandle_t handle,
          cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n,
          const float* alpha,
          const float* A, int lda,
          const float* beta,
          const float* B, int ldb,
          float* C, int ldc
      ) {
    return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  }
  
  template <>
  cublasStatus_t geam(cublasHandle_t handle,
          cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n,
          const double* alpha,
          const double* A, int lda,
          const double* beta,
          const double* B, int ldb,
          double* C, int ldc
      ) {
    return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  }

  // 2D transpose
  template <class InputView, class OutputView,
        std::enable_if_t<InputView::rank()==2 && OutputView::rank()==2, std::nullptr_t> = nullptr>
  void transpose(const InputView& in, OutputView& out) {
    static_assert( std::is_same_v<typename InputView::value_type, typename OutputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename OutputView::layout_type> );
    assert( in.extent(0) == out.extent(1) );
    assert( in.extent(1) == out.extent(0) );
  
    using value_type = InputView::value_type;
    constexpr value_type alpha = 1;
    constexpr value_type beta = 0;
    // transpose by cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
  
    geam(handle,
         CUBLAS_OP_T,
         CUBLAS_OP_T,
         in.extent(1),
         in.extent(0),
         &alpha,
         in.data_handle(),
         in.extent(0),
         &beta,
         in.data_handle(),
         in.extent(0),
         out.data_handle(),
         out.extent(0)
        );
  
    cublasDestroy(handle);
  }

  template <class InputView, class OutputView,
          std::enable_if_t<InputView::rank()==3 && OutputView::rank()==3, std::nullptr_t> = nullptr>
  void transpose(const InputView& in, OutputView& out, const std::array<int, 3>& axes) {
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
      std::copy(std::execution::par_unseq, in.data_handle(), in.data_handle()+n, out.data_handle());
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
      transpose(sub_in, sub_out);
    } else if(axes == axes_type({2, 0, 1})) {
      using mdspan2d_type = stdex::mdspan<value_type, stdex::dextents<size_type, 2>, layout_type>;
      using extent2d_type = std::array<std::size_t, 2>;
      extent2d_type in_shape({in.extent(0) * in.extent(1), in.extent(2)});
      extent2d_type out_shape({in.extent(2), in.extent(0) * in.extent(1)});

      const mdspan2d_type sub_in(in.data_handle(), in_shape);
      mdspan2d_type sub_out(out.data_handle(), out_shape);
      transpose(sub_in, sub_out);
    } else if(axes == axes_type({2, 1, 0})) {
      Impl::for_each(policy3d,
        [=](const int i0, const int i1, const int i2) {
          out(i2, i1, i0) = in(i0, i1, i2);
      });
    } else {
      std::runtime_error("Invalid axes specified.");
    }
  }
};

#endif
