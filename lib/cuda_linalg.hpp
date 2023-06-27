#ifndef __CUDA_LINALG_HPP__
#define __CUDA_LINALG_HPP__

#include <cassert>
#include <thrust/device_vector.h>
#include <experimental/mdspan>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

static inline std::string
cusolverGetErrorString(cusolverStatus_t error) {
    switch (error) {
        #define CASE(x) case x: return #x;
        CASE(CUSOLVER_STATUS_SUCCESS);
        CASE(CUSOLVER_STATUS_NOT_INITIALIZED);
        CASE(CUSOLVER_STATUS_ALLOC_FAILED);
        CASE(CUSOLVER_STATUS_INVALID_VALUE);
        CASE(CUSOLVER_STATUS_ARCH_MISMATCH);
        CASE(CUSOLVER_STATUS_MAPPING_ERROR);
        CASE(CUSOLVER_STATUS_EXECUTION_FAILED);
        CASE(CUSOLVER_STATUS_INTERNAL_ERROR);
        CASE(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        CASE(CUSOLVER_STATUS_NOT_SUPPORTED);
        CASE(CUSOLVER_STATUS_ZERO_PIVOT);
        CASE(CUSOLVER_STATUS_INVALID_LICENSE);
        #undef CASE
    }

    return std::string("unknown error: `") + std::to_string(int(error)) + "`";
}

#define CUSOLVER_SAFE_CALL_FAILED( error ) \
  { \
    throw STD_RUNTIME_ERROR(std::string("CUSOLVER failed: ") + cusolverGetErrorString(error)); \
  }

#define CUSOLVER_SAFE_CALL( ... ) \
  { \
    cusolverStatus_t error = __VA_ARGS__; \
    if(error != CUSOLVER_STATUS_SUCCESS) { \
      CUSOLVER_SAFE_CALL_FAILED(error); \
    } \
  }

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

  template <typename T>
  cublasStatus_t gemmStridedBatched(cublasHandle_t handle,
          cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n, int k,
          const T* alpha,
          const T* A, int lda, long long int strideA,
          const T* B, int ldb, long long int strideB,
          const T* beta,
          T* C, int ldc, long long int strideC,
          int batchCount
      );
  
  template <>
  cublasStatus_t gemmStridedBatched(cublasHandle_t handle,
          cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n, int k,
          const float* alpha,
          const float* A, int lda, long long int strideA,
          const float* B, int ldb, long long int strideB,
          const float* beta,
          float* C, int ldc, long long int strideC,
          int batchCount) {
  
    return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
  }
  
  template <>
  cublasStatus_t gemmStridedBatched(cublasHandle_t handle,
          cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n, int k,
          const double* alpha,
          const double* A, int lda, long long int strideA,
          const double* B, int ldb, long long int strideB,
          const double* beta,
          double* C, int ldc, long long int strideC,
          int batchCount) {
  
    return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
  }

  template <typename T>
  inline cusolverStatus_t syevjBatched_bufferSize(cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        cublasFillMode_t uplo,
        int n,
        const T* A,
        int lda,
        const T* W,
        int* lwork,
        syevjInfo_t params,
        int batchSize
  );

  template <>
  cusolverStatus_t syevjBatched_bufferSize(
          cusolverDnHandle_t handle,
          cusolverEigMode_t jobz,
          cublasFillMode_t uplo,
          int n,
          const float* a,
          int lda,
          const float* w,
          int* lwork,
          syevjInfo_t params,
          int batchSize) {
    return cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, a, lda, w, lwork, params, batchSize);
  }
  
  template <>
  cusolverStatus_t syevjBatched_bufferSize(
          cusolverDnHandle_t handle,
          cusolverEigMode_t jobz,
          cublasFillMode_t uplo,
          int n,
          const double* a,
          int lda,
          const double* w,
          int* lwork,
          syevjInfo_t params,
          int batchSize) {
    return cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, a, lda, w, lwork, params, batchSize);
  }

  template<typename T>
  inline cusolverStatus_t syevjBatched(
          cusolverDnHandle_t handle,
          cusolverEigMode_t jobz,
          cublasFillMode_t uplo,
          int n,
          T* A,
          int lda,
          T* W,
          T* work,
          int lwork,
          int* info,
          syevjInfo_t params,
          int batchSize
      );
  
  template<>
  inline cusolverStatus_t syevjBatched(
          cusolverDnHandle_t handle,
          cusolverEigMode_t jobz,
          cublasFillMode_t uplo,
          int n,
          float* A,
          int lda,
          float* W,
          float* work,
          int lwork,
          int* info,
          syevjInfo_t params,
          int batchSize) {
    return cusolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize);
  }
  
  template<>
  inline cusolverStatus_t syevjBatched(
          cusolverDnHandle_t handle,
          cusolverEigMode_t jobz,
          cublasFillMode_t uplo,
          int n,
          double* A,
          int lda,
          double* W,
          double* work,
          int lwork,
          int* info,
          syevjInfo_t params,
          int batchSize) {
    return cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize);
  }

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
    cublasHandle_t handle;
    cublasCreate(&handle);
  
    cublasOperation_t transa = _transa == "N" ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transb = _transb == "N" ? CUBLAS_OP_N : CUBLAS_OP_T;
  
    const auto Cm = C.extent(0);
    const auto Am = _transa == "N" ? A.extent(0) : A.extent(1);
    assert(Cm == Am);
  
    const auto Cn = C.extent(1);
    const auto Bn = _transb == "N" ? B.extent(1) : B.extent(0);
    assert(Cn == Bn);
  
    const auto Ak = _transa == "N" ? A.extent(1) : A.extent(0);
    const auto Bk = _transb == "N" ? B.extent(0) : B.extent(1);
    assert(Ak == Bk);
  
    auto status = gemmStridedBatched(handle,
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
                                    );
    cublasDestroy(handle);
  }
  
  /*
   * Batched matrix vector product
   * Matrix shape
   * A (n, m, l), B (m, l), C (n, l)
   * C = A * B
   * */
  template <class ViewA, class ViewB, class ViewC,
            std::enable_if_t<ViewA::rank()==3 && ViewB::rank()==2 && ViewC::rank()==2, std::nullptr_t> = nullptr>
  void matrix_vector_product(const ViewA& A,
                             const ViewB& B,
                             ViewC& C,
                             std::string _transa,
                             typename ViewA::value_type alpha = 1
                            ) {
    cublasHandle_t handle;
    cublasCreate(&handle);
  
    cublasOperation_t transa = _transa == "N" ? CUBLAS_OP_N : CUBLAS_OP_T;
  
    const auto Cm = C.extent(0);
    const auto Am = _transa == "N" ? A.extent(0) : A.extent(1);
    assert(Cm == Am);
  
    const auto Ak = _transa == "N" ? A.extent(1) : A.extent(0);
    const auto Bk = B.extent(0);
    assert(Ak == Bk);
    
    using value_type = ViewA::value_type; 
    const value_type beta = 0;
    auto status = gemmStridedBatched(handle,
                                     transa,
                                     CUBLAS_OP_N,
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
                                    );
    cublasDestroy(handle);
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
  
    using value_type = MatrixView::value_type;
    assert(a.extent(0) == v.extent(0));
    assert(a.extent(0) == a.extent(1)); // Square array
    assert(a.extent(2) == v.extent(1)); // batch size
  
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
  
    syevjInfo_t params;
    constexpr value_type tol = 1.0e-7;
    constexpr int max_sweeps = 100;
    constexpr int sort_eig = 0;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  
    cusolverDnCreateSyevjInfo(&params);
    cusolverDnXsyevjSetTolerance(params, tol);
    cusolverDnXsyevjSetMaxSweeps(params, max_sweeps);
    cusolverDnXsyevjSetSortEig(params, sort_eig);

    const int batchSize = v.extent(1);
    int lwork = 0;
    syevjBatched_bufferSize(
                      handle,
                      jobz,
                      uplo,
                      a.extent(0), a.data_handle(),
                      v.extent(0), v.data_handle(),
                      &lwork,
                      params,
                      batchSize
                     );
  
    thrust::device_vector<value_type> workspace(lwork);
    thrust::device_vector<int> info(batchSize, 0);
    value_type* workspace_data = (value_type *)thrust::raw_pointer_cast(workspace.data());
    int* info_data = (int *)thrust::raw_pointer_cast(info.data());
  
    auto status = syevjBatched(
                   handle,
                   jobz,
                   uplo,
                   a.extent(0), a.data_handle(),
                   v.extent(0), v.data_handle(),
                   workspace_data, workspace.size(),
                   info_data,
                   params,
                   batchSize
                  );

    cudaDeviceSynchronize();
    cusolverDnDestroy(handle);
  }

  // 2D transpose
  template <class InputView, class OutputView,
        std::enable_if_t<InputView::rank()==2 && OutputView::rank()==2, std::nullptr_t> = nullptr>
  void transpose(const InputView& in, OutputView& out) {
    static_assert( std::is_same_v<typename InputView::value_type, typename OutputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename OutputView::layout_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, stdex::layout_left> );

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
};

#endif
