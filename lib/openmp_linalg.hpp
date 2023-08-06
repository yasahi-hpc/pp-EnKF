#ifndef __OPENMP_LINALG_HPP__
#define __OPENMP_LINALG_HPP__

#include <cassert>
#include <experimental/mdspan>
#include <Eigen/Dense>

namespace stdex = std::experimental;

namespace Impl {
  struct blasHandle_t {
  public:
    void create() {}
    void destroy() {}
  };

  template <class T>
  struct syevjHandle_t {
  public:
    template <class MatrixView, class VectorView,
              std::enable_if_t<MatrixView::rank()==3 && VectorView::rank()==2, std::nullptr_t> = nullptr>
    void create(MatrixView& a, VectorView& v, T tol=1.0e-7, int max_sweeps=100, int sort_eig=0) {}
    void destroy() {}
  };

  /*
   * Batched matrix matrix product
   * Matrix shape
   * A (n, m, l), B (m, k, l), C (n, k, l)
   * C = alpha * Op(A) * Op(B) + beta * C
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

    static_assert( std::is_same_v<typename ViewA::value_type, typename ViewB::value_type> );
    static_assert( std::is_same_v<typename ViewA::value_type, typename ViewC::value_type> );
    static_assert( std::is_same_v<typename ViewA::layout_type, typename ViewB::layout_type> );
    static_assert( std::is_same_v<typename ViewA::layout_type, typename ViewC::layout_type> );
    static_assert( std::is_same_v<typename ViewA::layout_type, stdex::layout_left>, "The input Layout must be LayoutLeft");
  
    const auto Cm = C.extent(0);
    const auto Am = _transa == "N" ? A.extent(0) : A.extent(1);
    assert(Cm == Am);
  
    const auto Cn = C.extent(1);
    const auto Bn = _transb == "N" ? B.extent(1) : B.extent(0);
    assert(Cn == Bn);
  
    const auto Ak = _transa == "N" ? A.extent(1) : A.extent(0);
    const auto Bk = _transb == "N" ? B.extent(0) : B.extent(1);
    assert(Ak == Bk);

    using value_type = ViewA::value_type;
    using matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>;

    const int batchSize = C.extent(2);

    #pragma omp parallel for
    for(int ib=0; ib<batchSize; ib++) {
      auto sub_A = stdex::submdspan(A, std::full_extent, std::full_extent, ib);
      auto sub_B = stdex::submdspan(B, std::full_extent, std::full_extent, ib);
      auto sub_C = stdex::submdspan(C, std::full_extent, std::full_extent, ib);

      Eigen::Map<const matrix_type> matA(sub_A.data_handle(), sub_A.extent(0), sub_A.extent(1));
      Eigen::Map<const matrix_type> matB(sub_B.data_handle(), sub_B.extent(0), sub_B.extent(1));
      Eigen::Map<matrix_type> matC(sub_C.data_handle(), sub_C.extent(0), sub_C.extent(1));

      if(_transa == "N" && _transb == "N") {
        matC.noalias() = alpha * matA * matB + beta * matC;
      } else if(_transa == "N" && _transb == "T") {
        matC.noalias() = alpha * matA * matB.transpose() + beta * matC;
      } else if(_transa == "T" && _transb == "N") {
        matC.noalias() = alpha * matA.transpose() * matB + beta * matC;
      } else {
        matC.noalias() = alpha * matA.transpose() * matB.transpose() + beta * matC;
      }
    }
  }

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
    matrix_matrix_product(A, B, C, _transa, _transb, alpha, beta);
  }
  
  /*
   * Batched matrix vector product
   * Matrix shape
   * A (n, m, l), B (m, l), C (n, l)
   * C = alpha * Op(A) * B
   * */
  template <class ViewA, class ViewB, class ViewC,
            std::enable_if_t<ViewA::rank()==3 && ViewB::rank()==2 && ViewC::rank()==2, std::nullptr_t> = nullptr>
  void matrix_vector_product(const ViewA& A,
                             const ViewB& B,
                             ViewC& C,
                             std::string _transa,
                             typename ViewA::value_type alpha = 1
                            ) {
    static_assert( std::is_same_v<typename ViewA::value_type, typename ViewB::value_type> );
    static_assert( std::is_same_v<typename ViewA::value_type, typename ViewC::value_type> );
    static_assert( std::is_same_v<typename ViewA::layout_type, typename ViewB::layout_type> );
    static_assert( std::is_same_v<typename ViewA::layout_type, typename ViewC::layout_type> );
    static_assert( std::is_same_v<typename ViewA::layout_type, stdex::layout_left>, "The input Layout must be LayoutLeft");

    const auto Cm = C.extent(0);
    const auto Am = _transa == "N" ? A.extent(0) : A.extent(1);
    assert(Cm == Am);
  
    const auto Ak = _transa == "N" ? A.extent(1) : A.extent(0);
    const auto Bk = B.extent(0);
    assert(Ak == Bk);
    
    using value_type = ViewA::value_type; 
    using matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_type = Eigen::Vector<value_type, Eigen::Dynamic>;

    const int batchSize = C.extent(1);

    #pragma omp parallel for
    for(int ib=0; ib<batchSize; ib++) {
      auto sub_A = stdex::submdspan(A, std::full_extent, std::full_extent, ib);
      auto sub_B = stdex::submdspan(B, std::full_extent, ib);
      auto sub_C = stdex::submdspan(C, std::full_extent, ib);

      Eigen::Map<const matrix_type> matA(sub_A.data_handle(), sub_A.extent(0), sub_A.extent(1));
      Eigen::Map<const vector_type> vecB(sub_B.data_handle(), sub_B.extent(0));
      Eigen::Map<vector_type>       vecC(sub_C.data_handle(), sub_C.extent(0));

      if(_transa == "N") {
        vecC.noalias() = alpha * matA * vecB;
      } else {
        vecC.noalias() = alpha * matA.transpose() * vecB;
      }
    }
  }

  template <class ViewA, class ViewB, class ViewC,
            std::enable_if_t<ViewA::rank()==3 && ViewB::rank()==2 && ViewC::rank()==2, std::nullptr_t> = nullptr>
  void matrix_vector_product(const blasHandle_t& blas_handle,
                             const ViewA& A,
                             const ViewB& B,
                             ViewC& C,
                             std::string _transa,
                             typename ViewA::value_type alpha = 1
                            ) {
    matrix_vector_product(A, B, C, _transa, alpha);
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
    static_assert( std::is_same_v<typename MatrixView::layout_type, stdex::layout_left>, "The input Layout must be LayoutLeft");
  
    assert(a.extent(0) == v.extent(0));
    assert(a.extent(0) == a.extent(1)); // Square array
    assert(a.extent(2) == v.extent(1)); // batch size

    using value_type = MatrixView::value_type;
    using vector_type = Eigen::Vector<value_type, Eigen::Dynamic>;
    using matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>;

    const int batchSize = v.extent(1);
    #pragma omp parallel for
    for(int ib=0; ib<batchSize; ib++) {
      auto sub_a = stdex::submdspan(a, std::full_extent, std::full_extent, ib);
      auto sub_v = stdex::submdspan(v, std::full_extent, ib);

      Eigen::Map<matrix_type> mat(sub_a.data_handle(), sub_a.extent(0), sub_a.extent(1));
      Eigen::Map<vector_type> vec(sub_v.data_handle(), sub_v.extent(0));
      
      Eigen::SelfAdjointEigenSolver<matrix_type> eigensolver(mat);
      mat.noalias() = eigensolver.eigenvectors();
      vec.noalias() = eigensolver.eigenvalues();
    }
  }

  template <class Handle, class MatrixView, class VectorView,
            std::enable_if_t<MatrixView::rank()==3 && VectorView::rank()==2, std::nullptr_t> = nullptr>
  void eig(const Handle& syevj_handle, MatrixView& a, VectorView& v) {
    eig(a, v);
  }

  // 2D transpose
  template <class InputView, class OutputView, int _blocksize=16,
            std::enable_if_t<InputView::rank()==2 && OutputView::rank()==2, std::nullptr_t> = nullptr>
  void transpose(const InputView& in, OutputView& out) {
    static_assert( std::is_same_v<typename InputView::value_type, typename OutputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename OutputView::layout_type> );

    assert( in.extent(0) == out.extent(1) );
    assert( in.extent(1) == out.extent(0) );
    using layout_type = InputView::layout_type;
    int row, col;
    if(std::is_same_v<layout_type, stdex::layout_left>) {
      row = in.extent(0);
      col = in.extent(1);
    } else {
      row = in.extent(1);
      col = in.extent(0);
    }

    const int blocksize = (row >= _blocksize && col >= _blocksize) ? _blocksize : std::min(row, col);

    #pragma omp parallel for collapse(2)
    for(int j=0; j<col; j+=blocksize) {
      for(int i=0; i<row; i+=blocksize) {
        for(int c=j; c<j+blocksize && c<col; c++) {
          for(int r=i; r<i+blocksize && r<row; r++) {
            out(c, r) = in(r, c);
          }
        }
      }
    }
  }

  template <class InputView, class OutputView, int _blocksize=16,
            std::enable_if_t<InputView::rank()==2 && OutputView::rank()==2, std::nullptr_t> = nullptr>
  void transpose(const blasHandle_t& blas_handle, const InputView& in, OutputView& out) {
    transpose(in, out);
  }
};

#endif
