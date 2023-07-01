#ifndef __LETKF_SOLVER_HPP__
#define __LETKF_SOLVER_HPP__

#include <tuple>
#include <stdpar/Parallel_For.hpp>
#include <stdpar/numpy_like.hpp>
#include <linalg.hpp>
#include "types.hpp"
#include "../mpi_config.hpp"

using letkf_config_type = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t, double>;

class LETKFSolver {
private:
  using value_type = RealView2D::value_type;
  Impl::blasHandle_t blas_handle_;
  Impl::syevjHandle_t<value_type> syevj_handle_;
  
  RealView3D X_, dX_; // (n_stt, n_ens, n_batch)
  RealView3D Y_, dY_; // (n_obs, n_ens, n_batch)

  RealView3D x_mean_; // (n_stt, 1, n_batch)
  RealView3D y_mean_; // (n_obs, 1, n_batch)

  RealView3D yo_; // (n_obs, 1, n_batch)

  RealView3D I_; // (n_ens, n_ens, n_batch)
  RealView3D Q_; // (n_ens, n_ens, n_batch)
  RealView3D V_; // (n_ens, n_ens, n_batch)
  RealView2D d_; // (n_ens, n_batch)
  RealView3D inv_D_; // (n_ens, n_ens, n_batch)
  RealView3D P_; // (n_ens, n_ens, n_batch)
 
  RealView3D rR_; // (n_obs, n_obs, n_batch) (rho o R)^-1
  RealView3D w_; // (n_ens, 1, n_batch)
  RealView3D W_; // (n_ens, n_ens, n_batch)

  // Buffers
  RealView3D tmp_ee_; // (n_ens, n_ens, n_batch)
  RealView3D tmp_oe_; // (n_obs, n_ens, n_batch)
  RealView2D tmp_o_; // (n_obs, n_batch)
  RealView2D tmp_e_; // (n_ens, n_batch)

  std::size_t n_ens_;
  std::size_t n_stt_;
  std::size_t n_obs_;
  std::size_t n_batch_;

  value_type beta_; // covariance inflation

public:
  LETKFSolver(const letkf_config_type& letkf_config) {
    n_ens_   = std::get<0>(letkf_config);
    n_stt_   = std::get<1>(letkf_config);
    n_obs_   = std::get<2>(letkf_config);
    n_batch_ = std::get<3>(letkf_config);
    beta_    = static_cast<value_type>( std::get<4>(letkf_config) );

    // Allocate views
    initialize();
  }

  ~LETKFSolver(){
    blas_handle_.destroy();
    syevj_handle_.destroy();
  }

public:
  // Getters
  auto& X() { return X_; }
  auto& Y() { return Y_; }
  auto& y_obs() { return yo_; }
  auto& rR() { return rR_; }

public:
  void solve() {
    auto X = X_.mdspan();
    auto Y = Y_.mdspan();
    auto dX = dX_.mdspan();
    auto dY = dY_.mdspan();
    auto yo = yo_.mdspan();
    auto x_mean = x_mean_.mdspan();
    auto y_mean = y_mean_.mdspan();

    // Ensemble average
    Impl::mean(X, x_mean, 1); // (n_stt, n_ens, n_batch) -> (n_stt, 1, n_batch)
    Impl::mean(Y, y_mean, 1); // (n_obs, n_ens, n_batch) -> (n_obs, 1, n_batch)

    // dX = X - mean(X), dY = Y - mean(Y)
    Impl::axpy(X, x_mean, dX, -1); // (n_stt, n_ens, n_batch) - (n_stt, 1, n_batch) -> (n_stt, n_ens, n_batch)
    Impl::axpy(Y, y_mean, dY, -1); // (n_obs, n_ens, n_batch) - (n_obs, 1, n_batch) -> (n_obs, n_ens, n_batch)

    // dyo = yo - mean(Y)
    Impl::axpy(yo, y_mean, -1); // (n_obs, 1, n_batch)

    // Q = (Ne-1)I/beta + dY^T * rR * dY
    auto rR = rR_.mdspan(); // (n_obs, n_obs, n_batch)
    auto I = I_.mdspan();
    auto Q = Q_.mdspan();
    auto tmp_oe = tmp_oe_.mdspan();
    const value_type beta = (static_cast<int>(n_ens_) - 1) / beta_;
    Impl::deep_copy(I, Q); // (n_ens, n_ens, n_batch)
    Impl::matrix_matrix_product(blas_handle_, rR, dY, tmp_oe, "N", "N"); // (n_obs, n_obs, n_batch) * (n_obs, n_ens, n_batch) -> (n_obs, n_ens, n_batch)
    Impl::matrix_matrix_product(blas_handle_, dY, tmp_oe, Q, "T", "N", 1, beta); // (n_ens, n_obs, n_batch) * (n_obs, n_ens, n_batch) -> (n_ens, n_ens, n_batch)

    // Q = V * diag(d) * V^T
    auto d = d_.mdspan();
    auto V = V_.mdspan();
    Impl::deep_copy(Q, V);
    Impl::eig(syevj_handle_, V, d); // (n_ens, n_ens, n_batch) -> (n_ens, n_ens, n_batch), (n_ens, n_batch)

    // P = V * inv(d) * V^T
    // P: (n_ens, n_ens, n_batch)
    auto inv_D = inv_D_.mdspan();
    auto tmp_ee = tmp_ee_.mdspan();
    auto P = P_.mdspan();
    Impl::diag(d, inv_D, -1); // (n_ens, n_ens, n_batch)
    Impl::matrix_matrix_product(blas_handle_, inv_D, V, tmp_ee, "N", "T"); // (n_ens, n_ens, n_batch) * (n_ens, n_ens, n_batch) -> (n_ens, n_ens, n_batch)
    Impl::matrix_matrix_product(blas_handle_, V, tmp_ee, P, "N", "N"); // (n_ens, n_ens, n_batch) * (n_ens, n_ens, n_batch) -> (n_ens, n_ens, n_batch)

    // w = P * (dY^T * inv(R) * dyo)
    auto w  = w_.mdspan();
    auto tmp_o = tmp_o_.mdspan();
    auto tmp_e = tmp_e_.mdspan();
    auto dyo = Impl::squeeze(yo, 1);
    auto _w = Impl::squeeze(w, 1);
    Impl::matrix_vector_product(blas_handle_, rR, dyo, tmp_o, "N"); // (n_obs, n_obs, n_batch) * (n_obs, n_batch) -> (n_obs, n_batch)
    Impl::matrix_vector_product(blas_handle_, dY, tmp_o, tmp_e, "T"); // (n_ens, n_obs, n_batch) * (n_obs, n_batch) -> (n_ens, n_batch)
    Impl::matrix_vector_product(blas_handle_, P, tmp_e, _w, "N"); // (n_ens, n_ens, n_batch) * (n_ens, n_batch) -> (n_ens, n_batch)

    // W = sqrt(Ne-1) * V * inv(sqrt(D)) * V^T
    auto W = W_.mdspan();
    const value_type alpha = sqrt(static_cast<int>(n_ens_) - 1);
    Impl::diag(d, inv_D, -0.5); // (n_ens, n_ens, n_batch)
    Impl::matrix_matrix_product(blas_handle_, inv_D, V, tmp_ee, "N", "T"); // (n_ens, n_ens, n_batch) * (n_ens, n_ens, n_batch) -> (n_ens, n_ens, n_batch)
    Impl::matrix_matrix_product(blas_handle_, V, tmp_ee, W, "N", "N", alpha); // (n_ens, n_ens, n_batch) * (n_ens, n_ens, n_batch) -> (n_ens, n_ens, n_batch)

    // W = W + w
    // Xsol = x_mean + matmat(dX, W)
    Impl::axpy(W, w); // (n_ens, n_ens, n_batch) + (n_ens, 1, n_batch) -> (n_ens, n_ens, n_batch)
    Impl::matrix_matrix_product(blas_handle_, dX, W, X, "N", "N"); // (n_stt, n_ens, n_batch) * (n_ens, n_ens, n_batch) -> (n_stt, n_ens, n_batch)
    Impl::axpy(X, x_mean); // (n_stt, n_ens, n_batch) + (n_stt, 1, n_batch) -> (n_stt, n_ens, n_batch)
  }

private:
  void initialize() {
    // Allocate Views
    X_  = RealView3D("X", n_stt_, n_ens_, n_batch_);
    dX_ = RealView3D("dX", n_stt_, n_ens_, n_batch_);
    Y_  = RealView3D("Y", n_obs_, n_ens_, n_batch_);
    dY_ = RealView3D("dY", n_obs_, n_ens_, n_batch_);

    x_mean_ = RealView3D("x_mean", n_stt_, 1, n_batch_);
    y_mean_ = RealView3D("y_mean", n_obs_, 1, n_batch_);

    yo_ = RealView3D("yo", n_obs_, 1, n_batch_);

    I_     = RealView3D("I", n_ens_, n_ens_, n_batch_);
    Q_     = RealView3D("Q", n_ens_, n_ens_, n_batch_);
    V_     = RealView3D("V", n_ens_, n_ens_, n_batch_);
    d_     = RealView2D("d", n_ens_, n_batch_);
    inv_D_ = RealView3D("inv_D", n_ens_, n_ens_, n_batch_);
    P_     = RealView3D("P", n_ens_, n_ens_, n_batch_);

    rR_ = RealView3D("rR", n_obs_, n_obs_, n_batch_);
    w_  = RealView3D("w", n_ens_, 1, n_batch_);
    W_  = RealView3D("W", n_ens_, n_ens_, n_batch_);

    tmp_ee_ = RealView3D("tmp_ee", n_ens_, n_ens_, n_batch_);
    tmp_oe_ = RealView3D("tmp_oe", n_obs_, n_ens_, n_batch_);
    tmp_o_  = RealView2D("tmp_o", n_obs_, n_batch_);
    tmp_e_  = RealView2D("tmp_e", n_ens_, n_batch_);

    auto rR = rR_.mdspan();
    auto I = I_.mdspan();
    Impl::identity(rR);
    Impl::identity(I);

    auto d = d_.mdspan();
    auto V = V_.mdspan();
    blas_handle_.create();
    syevj_handle_.create(V, d);
  }
};

#endif
