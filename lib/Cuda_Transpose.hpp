#ifndef __CUDA_TRANSPOSE_HPP__
#define __CUDA_TRANSPOSE_HPP__

#include <cublas_v2.h>
#include <experimental/mdspan>

namespace stdex = std::experimental;

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
    #include <thrust/complex.h>
    template <typename RealType> using Complex = thrust::complex<RealType>;
#else
    #include <complex>
    template <typename RealType> using Complex = std::complex<RealType>;
#endif

namespace Impl {
  template <typename ScalarType, class LayoutPolicy = stdex::layout_left,
            typename std::enable_if<std::is_same<ScalarType, float           >::value ||
                                    std::is_same<ScalarType, double          >::value ||
                                    std::is_same<ScalarType, Complex<float>  >::value ||
                                    std::is_same<ScalarType, Complex<double> >::value
                                    , std::nullptr_t>::type = nullptr
  >
  struct Transpose {
    private:
      int col_;
      int row_;
      cublasHandle_t handle_;

    public:
      using array_layout = LayoutPolicy;
     
    public:
      Transpose() = delete;
    
      Transpose(int row, int col) : row_(row), col_(col) {
        if(std::is_same<array_layout, stdex::layout_right>::value) {
          row_ = col;
          col_ = row;
        }
        cublasCreate(&handle_);
      }

      ~Transpose() {
        cublasDestroy(handle_);
      }

      // Out-place transpose
      void forward(ScalarType* dptr_in, ScalarType* dptr_out) {
        cublasTranspose_(dptr_in, dptr_out, row_, col_);
      }

      void backward(ScalarType* dptr_in, ScalarType* dptr_out) {
        cublasTranspose_(dptr_in, dptr_out, col_, row_);
      }

    private:
      // float32 specialization
      template <typename RType=ScalarType,
                typename std::enable_if<std::is_same<RType, float>::value, std::nullptr_t>::type = nullptr>
      void cublasTranspose_(RType* dptr_in, RType* dptr_out, int row, int col) {
        constexpr float alpha = 1.0;
        constexpr float beta  = 0.0;
        cublasSgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_T, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    dptr_in,     // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    dptr_in,     // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    dptr_out,    // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // float64 specialization
      template <typename RType=ScalarType,
                typename std::enable_if<std::is_same<RType, double>::value, std::nullptr_t>::type = nullptr>
      void cublasTranspose_(RType* dptr_in, RType* dptr_out, int row, int col) {
        constexpr double alpha = 1.;
        constexpr double beta  = 0.;
        cublasDgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_T, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    dptr_in,     // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    dptr_in,     // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    dptr_out,    // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // complex64 specialization
      template <typename CType=ScalarType,
                typename std::enable_if<std::is_same<CType, Complex<float> >::value, std::nullptr_t>::type = nullptr>
      void cublasTranspose_(CType* dptr_in, CType* dptr_out, int row, int col) {
        const cuComplex alpha = make_cuComplex(1.0, 0.0);
        const cuComplex beta  = make_cuComplex(0.0, 0.0);
        cublasCgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_N, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    reinterpret_cast<cuComplex*>(dptr_in), // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    reinterpret_cast<cuComplex*>(dptr_in), // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    reinterpret_cast<cuComplex*>(dptr_out), // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // complex128 specialization
      template <typename CType=ScalarType,
                typename std::enable_if<std::is_same<CType, Complex<double> >::value, std::nullptr_t>::type = nullptr>
      void cublasTranspose_(CType* dptr_in, CType* dptr_out, int row, int col) {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1., 0.);
        const cuDoubleComplex beta  = make_cuDoubleComplex(0., 0.);
        cublasZgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_N, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    reinterpret_cast<cuDoubleComplex*>(dptr_in),  // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    reinterpret_cast<cuDoubleComplex*>(dptr_in),  // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    reinterpret_cast<cuDoubleComplex*>(dptr_out), // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

  };
};

#endif
