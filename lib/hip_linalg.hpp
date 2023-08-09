#ifndef __HIP_LINALG_HPP__
#define __HIP_LINALG_HPP__

#include <cassert>
#include <thrust/device_vector.h>
#include <experimental/mdspan>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <type_traits>
#include "HIP_Helper.hpp"

namespace Impl {
  template <typename T>
  inline rocblas_status geam(rocblas_handle handle,
                      rocblas_operation transa,
                      rocblas_operation transb,
                      int m, int n,
                      const T* alpha,
                      const T* A, int lda,
                      const T* beta,
                      const T* B, int ldb,
                      T* C, int ldc
                     );
 
  template <>
  inline rocblas_status geam(rocblas_handle handle,
                      rocblas_operation transa,
                      rocblas_operation transb,
                      int m, int n,
                      const float* alpha,
                      const float* A, int lda,
                      const float* beta,
                      const float* B, int ldb,
                      float* C, int ldc
                     ) {
    return rocblas_sgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  }

  template <>
  inline rocblas_status geam(rocblas_handle handle,
                      rocblas_operation transa,
                      rocblas_operation transb,
                      int m, int n,
                      const double* alpha,
                      const double* A, int lda,
                      const double* beta,
                      const double* B, int ldb,
                      double* C, int ldc
                     ) {
    return rocblas_dgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  }

  template <typename T>
  inline rocblas_status gemmStridedBatched(rocblas_handle handle,
                                           rocblas_operation transa,
                                           rocblas_operation transb,
                                           int m, int n, int k,
                                           const T* alpha,
                                           const T* A, int lda, int strideA,
                                           const T* B, int ldb, int strideB,
                                           const T* beta,
                                           T* C,
                                           int ldc, int strideC,
                                           int batch_count
                                          );

  template <>
  inline rocblas_status gemmStridedBatched(rocblas_handle handle,
                                           rocblas_operation transa,
                                           rocblas_operation transb,
                                           int m, int n, int k,
                                           const float* alpha,
                                           const float* A, int lda, int strideA,
                                           const float* B, int ldb, int strideB,
                                           const float* beta,
                                           float* C,
                                           int ldc, int strideC,
                                           int batch_count
                                          ) {
    return rocblas_sgemm_strided_batched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_count);
  }

  template <>
  inline rocblas_status gemmStridedBatched(rocblas_handle handle,
                                           rocblas_operation transa,
                                           rocblas_operation transb,
                                           int m, int n, int k,
                                           const double* alpha,
                                           const double* A, int lda, int strideA,
                                           const double* B, int ldb, int strideB,
                                           const double* beta,
                                           double* C,
                                           int ldc, int strideC,
                                           int batch_count
                                          ) {
    return rocblas_dgemm_strided_batched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_count);
  }

  template <typename T>
  inline rocblas_status syevjStridedBatched(rocblas_handle handle,
                                            const rocblas_esort esort,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            T* A,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            const T abstol,
                                            T* residual,
                                            const rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            T* W,
                                            const rocblas_stride strideW,
                                            rocblas_int* info,
                                            const rocblas_int batch_count);

  template <>
  inline rocblas_status syevjStridedBatched(rocblas_handle handle,
                                            const rocblas_esort esort,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            float* A,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            const float abstol,
                                            float* residual,
                                            const rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            const rocblas_stride strideW,
                                            rocblas_int* info,
                                            const rocblas_int batch_count) {
    return rocsolver_ssyevj_strided_batched(handle, esort, evect, uplo, n, A, lda, strideA, abstol, residual, max_sweeps, n_sweeps, W, strideW, info, batch_count);
  }

  template <>
  inline rocblas_status syevjStridedBatched(rocblas_handle handle,
                                            const rocblas_esort esort,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            double* A,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            const double abstol,
                                            double* residual,
                                            const rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            const rocblas_stride strideW,
                                            rocblas_int* info,
                                            const rocblas_int batch_count) {
    return rocsolver_dsyevj_strided_batched(handle, esort, evect, uplo, n, A, lda, strideA, abstol, residual, max_sweeps, n_sweeps, W, strideW, info, batch_count);
  }

  template <typename T>
  inline rocblas_status syevdStridedBatched(rocblas_handle handle,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            T* A,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            T* D,
                                            const rocblas_stride strideD,
                                            T* E,
                                            const rocblas_stride strideE,
                                            rocblas_int* info,
                                            const rocblas_int batch_count);

  template <>
  inline rocblas_status syevdStridedBatched(rocblas_handle handle,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            float* A,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            float* D,
                                            const rocblas_stride strideD,
                                            float* E,
                                            const rocblas_stride strideE,
                                            rocblas_int* info,
                                            const rocblas_int batch_count) {
    return rocsolver_ssyevd_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
  }

  template <>
  inline rocblas_status syevdStridedBatched(rocblas_handle handle,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            double* A,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            double* D,
                                            const rocblas_stride strideD,
                                            double* E,
                                            const rocblas_stride strideE,
                                            rocblas_int* info,
                                            const rocblas_int batch_count) {
    return rocsolver_dsyevd_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
  }

  template <typename T>
  inline rocblas_status syevStridedBatched(rocblas_handle handle,
                                           const rocblas_evect evect,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           T* A,
                                           const rocblas_int lda,
                                           const rocblas_stride strideA,
                                           T* D,
                                           const rocblas_stride strideD,
                                           T* E,
                                           const rocblas_stride strideE,
                                           rocblas_int* info,
                                           const rocblas_int batch_count);

  template <>
  inline rocblas_status syevStridedBatched(rocblas_handle handle,
                                           const rocblas_evect evect,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           float* A,
                                           const rocblas_int lda,
                                           const rocblas_stride strideA,
                                           float* D,
                                           const rocblas_stride strideD,
                                           float* E,
                                           const rocblas_stride strideE,
                                           rocblas_int* info,
                                           const rocblas_int batch_count) {
    return rocsolver_ssyev_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
  }

  template <>
  inline rocblas_status syevStridedBatched(rocblas_handle handle,
                                           const rocblas_evect evect,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           double* A,
                                           const rocblas_int lda,
                                           const rocblas_stride strideA,
                                           double* D,
                                           const rocblas_stride strideD,
                                           double* E,
                                           const rocblas_stride strideE,
                                           rocblas_int* info,
                                           const rocblas_int batch_count) {
    return rocsolver_dsyev_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
  }

  struct blasHandle_t {
    rocblas_handle handle_;

  public:
    void create() {
      SafeHIPCall( rocblas_create_handle(&handle_) );
      SafeHIPCall( rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host) );
    }

    void destroy() {
      SafeHIPCall( rocblas_destroy_handle(handle_) );
    }
  };

  template <class T>
  struct syevjHandle_t {
    rocblas_handle handle_;
    thrust::device_vector<T> residual_vector_;
    thrust::device_vector<T> buffer_;
    thrust::device_vector<int> n_sweeps_vector_;
    thrust::device_vector<int> info_vector_;

    rocblas_esort esort_ = rocblas_esort_none;
    rocblas_evect evect_ = rocblas_evect_original;
    rocblas_fill uplo_ = rocblas_fill_lower;
    T abstol_;
    int max_sweeps_;

  public:
    template <class MatrixView, class VectorView,
              std::enable_if_t<MatrixView::rank()==3 && VectorView::rank()==2, std::nullptr_t> = nullptr>
    void create(MatrixView& a, VectorView& v, T tol=1.0e-7, int max_sweeps=100, int sort_eig=0) {
      SafeHIPCall( rocblas_create_handle(&handle_) );
      abstol_ = tol;
      max_sweeps_ = max_sweeps;

      int batch_count = v.extent(1);
      residual_vector_.resize(batch_count, 0);
      n_sweeps_vector_.resize(batch_count, 0);
      info_vector_.resize(batch_count, 0);

      // Used for rocsolver_ssyevd_strided_batched
      buffer_.resize(v.size(), 0);
    }

    void destroy() {
      SafeHIPCall( rocblas_destroy_handle(handle_) );
    }
  };

  /*
   * Batched matrix matrix product
   * Matrix shape
   * A (n, m, l), B (m, k, l), C (n, k, l)
   * */
   template <class ViewA, class ViewB, class ViewC,
             std::enable_if_t<ViewA::rank()==3 && ViewB::rank()==3 && ViewC::rank()==3, std::nullptr_t> = nullptr>
   void matrix_matrix_product(const ViewA& A,
                              const ViewB& B,
                              ViewC& C,
                              std::string _transa,
                              std::string _transb,
                              typename ViewA::value_type alpha = 1,
                              typename ViewA::value_type beta = 0) {
     Impl::blasHandle_t blas_handle;
     blas_handle.create();
     matrix_matrix_product(blas_handle, A, B, C, _transa, _transb, alpha, beta);
     blas_handle.destroy();
   }

  /*
   * Batched matrix matrix product
   * Matrix shape
   * A (n, m, l), B (m, k, l), C (n, k, l)
   * */
   template <class ViewA, class ViewB, class ViewC,
             std::enable_if_t<ViewA::rank()==3 && ViewB::rank()==3 && ViewC::rank()==3, std::nullptr_t> = nullptr>
   void matrix_matrix_product(const blasHandle_t& blas_handle,
                              const ViewA& A,
                              const ViewB& B,
                              ViewC& C,
                              std::string _transa,
                              std::string _transb,
                              typename ViewA::value_type alpha = 1,
                              typename ViewA::value_type beta = 0) {
    rocblas_operation transa = _transa == "N" ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation transb = _transb == "N" ? rocblas_operation_none : rocblas_operation_transpose;

    const auto Cm = C.extent(0);
    const auto Am = _transa == "N" ? A.extent(0) : A.extent(1);
    assert(Cm == Am);

    const auto Cn = C.extent(1);
    const auto Bn = _transb == "N" ? B.extent(1) : B.extent(0);
    assert(Cn == Bn);

    const auto Ak = _transa == "N" ? A.extent(1) : A.extent(0);
    const auto Bk = _transb == "N" ? B.extent(0) : B.extent(1);
    assert(Ak == Bk);

    SafeHIPCall(
      gemmStridedBatched(blas_handle.handle_,
                         transa,
                         transb,
                         Cm,
                         Cn,
                         Ak,
                         &alpha,
                         A.data_handle(),
                         A.extent(0),
                         A.extent(0) * A.extent(1),
                         B.data_handle(),
                         B.extent(0),
                         B.extent(0) * B.extent(1),
                         &beta,
                         C.data_handle(),
                         C.extent(0),
                         C.extent(0) * C.extent(1),
                         C.extent(2)
                        )
    );
  } 

  /*
   * Batched matrix vector product
   * Matrix shape
   * A (n, m, l), B (m, l), C (n, l)
   * C = A * B
   * */
   template <class ViewA, class ViewB, class ViewC,
             std::enable_if_t<ViewA::rank()==3 && ViewB::rank()==2 && ViewC::rank()==2, std::nullptr_t> = nullptr>
   inline void matrix_vector_product(const ViewA& A,
                                     const ViewB& B,
                                     ViewC& C,
                                     std::string _transa,
                                     typename ViewA::value_type alpha = 1
                                    ) {
     Impl::blasHandle_t blas_handle;
     blas_handle.create();
     matrix_vector_product(blas_handle, A, B, C, _transa, alpha);
     blas_handle.destroy();
   }

  /*
   * Batched matrix vector product
   * Matrix shape
   * A (n, m, l), B (m, l), C (n, l)
   * C = A * B
   * */
   template <class ViewA, class ViewB, class ViewC,
             std::enable_if_t<ViewA::rank()==3 && ViewB::rank()==2 && ViewC::rank()==2, std::nullptr_t> = nullptr>
   inline void matrix_vector_product(const blasHandle_t& blas_handle,
                                     const ViewA& A,
                                     const ViewB& B,
                                     ViewC& C,
                                     std::string _transa,
                                     typename ViewA::value_type alpha = 1
                                    ) {
    rocblas_operation transa = _transa == "N" ? rocblas_operation_none : rocblas_operation_transpose;

    const auto Cm = C.extent(0);
    const auto Am = _transa == "N" ? A.extent(0) : A.extent(1);
    assert(Cm == Am);

    const auto Ak = _transa == "N" ? A.extent(1) : A.extent(0);
    const auto Bk = B.extent(0);
    assert(Ak == Bk);

    using value_type = typename ViewA::value_type;
    const value_type beta = 0;

    SafeHIPCall(
      gemmStridedBatched(blas_handle.handle_,
                         transa,
                         rocblas_operation_none,
                         Cm,
                         1,
                         Ak,
                         &alpha,
                         A.data_handle(),
                         A.extent(0),
                         A.extent(0) * A.extent(1),
                         B.data_handle(),
                         B.extent(0),
                         B.extent(0),
                         &beta,
                         C.data_handle(),
                         C.extent(0),
                         C.extent(0),
                         C.extent(1)
                        )
    );
  }

  /*
   * Batched matrix eigen value decomposition (In-place only)
   * a @ v = w * v
   * a (m, m, l): square array
   * v (m, l)
   * w (m, m, l)
   * */
  template <class MatrixView, class VectorView,
            std::enable_if_t<MatrixView::rank()==3 && VectorView::rank()==2, std::nullptr_t> = nullptr>
  void eig(MatrixView& a, VectorView& v) {
    static_assert( std::is_same_v<typename MatrixView::value_type, typename VectorView::value_type> );
    static_assert( std::is_same_v<typename MatrixView::layout_type, typename VectorView::layout_type> );

    using value_type = typename MatrixView::value_type;
    Impl::syevjHandle_t<value_type> syevj_handle;
    syevj_handle.create(a, v);
    eig(syevj_handle, a, v);
    syevj_handle.destroy();
  }

  /*
   * Batched matrix eigen value decomposition (In-place only)
   * a @ v = w * v
   * a (m, m, l): square array
   * v (m, l)
   * w (m, m, l)
   * */
  template <class Handle, class MatrixView, class VectorView,
            std::enable_if_t<MatrixView::rank()==3 && VectorView::rank()==2, std::nullptr_t> = nullptr>
  void eig(const Handle& syevj_handle, MatrixView& a, VectorView& v) {
    static_assert( std::is_same_v<typename MatrixView::value_type, typename VectorView::value_type> );
    static_assert( std::is_same_v<typename MatrixView::layout_type, typename VectorView::layout_type> );

    using value_type = typename MatrixView::value_type;
    assert(a.extent(0) == v.extent(0));
    assert(a.extent(0) == a.extent(1)); // Square array
    assert(a.extent(2) == v.extent(1)); // batch size

    //rocblas_handle handle;
    //SafeHIPCall( rocblas_create_handle(&handle) );

    //rocblas_esort esort = rocblas_esort_none;
    //rocblas_evect evect = rocblas_evect_original;
    //rocblas_fill uplo = rocblas_fill_lower;
    //constexpr value_type abstol = 1.0e-7;
    //constexpr int max_sweeps = 100;
    //const int batch_count = v.extent(1);

    ////thrust::device_vector<value_type> residual_vector(batch_count, 0);
    //thrust::device_vector<int> n_sweeps_vector(batch_count, 0);
    //thrust::device_vector<int> info_vector(batch_count, 0);
    //value_type* residual = (value_type *)thrust::raw_pointer_cast(residual_vector.data());
    //int* n_sweeps = (int *)thrust::raw_pointer_cast(n_sweeps_vector.data());
    //int* info = (int *)thrust::raw_pointer_cast(info_vector.data());
    //
    //
    /*
    const int batch_count = v.extent(1);
    value_type* residual = (value_type *)thrust::raw_pointer_cast(syevj_handle.residual_vector_.data());
    int* n_sweeps = (int *)thrust::raw_pointer_cast(syevj_handle.n_sweeps_vector_.data());
    int* info = (int *)thrust::raw_pointer_cast(syevj_handle.info_vector_.data());

    SafeHIPCall(
      syevjStridedBatched(
        syevj_handle.handle_,
        syevj_handle.esort_,
        syevj_handle.evect_,
        syevj_handle.uplo_,
        a.extent(0),
        a.data_handle(),
        a.extent(0),
        a.extent(0) * a.extent(1),
        syevj_handle.abstol_,
        residual,
        syevj_handle.max_sweeps_,
        n_sweeps,
        v.data_handle(),
        v.extent(0),
        info,
        batch_count)
    );
    */

    //SafeHIPCall( rocblas_destroy_handle(handle) );
    //
    /*
    const int batch_count = v.extent(1);
    value_type* E = (value_type *)thrust::raw_pointer_cast(syevj_handle.buffer_.data());
    int* info = (int *)thrust::raw_pointer_cast(syevj_handle.info_vector_.data());
    SafeHIPCall(
      syevdStridedBatched(
        syevj_handle.handle_,
        syevj_handle.evect_,
        syevj_handle.uplo_,
        a.extent(0),
        a.data_handle(),
        a.extent(0),
        a.extent(0) * a.extent(1),
        v.data_handle(),
        v.extent(0),
        E,
        v.extent(0),
        info,
        batch_count)
    );
    */
    const int batch_count = v.extent(1);
    value_type* E = (value_type *)thrust::raw_pointer_cast(syevj_handle.buffer_.data());
    int* info = (int *)thrust::raw_pointer_cast(syevj_handle.info_vector_.data());
    SafeHIPCall(
      syevStridedBatched(
        syevj_handle.handle_,
        syevj_handle.evect_,
        syevj_handle.uplo_,
        a.extent(0),
        a.data_handle(),
        a.extent(0),
        a.extent(0) * a.extent(1),
        v.data_handle(),
        v.extent(0),
        E,
        v.extent(0),
        info,
        batch_count)
    );
  }

  // 2D transpose
  template <class InputView, class OutputView,
        std::enable_if_t<InputView::rank()==2 && OutputView::rank()==2, std::nullptr_t> = nullptr>
  inline void transpose(const blasHandle_t& blas_handle, const InputView& in, OutputView& out) {
    static_assert( std::is_same_v<typename InputView::value_type, typename OutputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename OutputView::layout_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, stdex::layout_left> );

    assert( in.extent(0) == out.extent(1) );
    assert( in.extent(1) == out.extent(0) );

    using value_type = typename InputView::value_type;
    constexpr value_type alpha = 1;
    constexpr value_type beta = 0;

    // transpose by rocblas
    SafeHIPCall(
     geam(blas_handle.handle_,
          rocblas_operation_transpose,
          rocblas_operation_transpose,
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
         )
    );
  }

  // 2D transpose
  template <class InputView, class OutputView,
        std::enable_if_t<InputView::rank()==2 && OutputView::rank()==2, std::nullptr_t> = nullptr>
  inline void transpose(const InputView& in, OutputView& out) {
    Impl::blasHandle_t blas_handle;
    blas_handle.create();
    transpose(blas_handle, in, out);
    blas_handle.destroy();
  }
};

#endif
