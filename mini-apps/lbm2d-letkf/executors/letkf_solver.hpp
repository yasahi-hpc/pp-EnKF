#ifndef __LETKF_SOLVER_HPP__
#define __LETKF_SOLVER_HPP__

#include <tuple>
#include <stdexec/execution.hpp>
#include <exec/on.hpp>
#include <executors/Parallel_For.hpp>
#include <executors/numpy_like.hpp>
#include <linalg.hpp>
#include "types.hpp"
#include "../mpi_config.hpp"

template <class Sender, class Scheduler, class InputView, class OutputView,
          std::enable_if_t<InputView::rank()==3 && OutputView::rank()==3, std::nullptr_t> = nullptr>
stdexec::sender auto mean_sender(Sender&& sender, Scheduler&& scheduler, const InputView& in, OutputView& out, int axis) {
  // 3D -> 3D (keepdims = true)
  int reduce_dim = axis==-1 ? InputView::rank()-1 : axis;
  assert(reduce_dim < InputView::rank());
  assert(out.extent(reduce_dim) == 1);
  const std::size_t reduce_size = in.extent(reduce_dim);
  const std::size_t n = in.size() / reduce_size;
  const auto n0 = (reduce_dim==0) ? in.extent(1) : in.extent(0);
  const auto n1 = n / n0;

  using value_type = InputView::value_type;

  return sender |
    exec::on(scheduler, stdexec::bulk(n,
      [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
        value_type sum = 0;
        const int i0 = idx%n0;
        const int i1 = idx/n0;
        for(int ir=0; ir < reduce_size; ir++) {
          if(reduce_dim == 0) {
            auto sub_in = submdspan(in, ir, std::experimental::full_extent, std::experimental::full_extent);
            sum += sub_in(i0, i1);
          } else if(reduce_dim == 1) {
            auto sub_in = submdspan(in, std::experimental::full_extent, ir, std::experimental::full_extent);
            sum += sub_in(i0, i1);
          } else {
            auto sub_in = submdspan(in, std::experimental::full_extent, std::experimental::full_extent, ir);
            sum += sub_in(i0, i1);
          }
        }
        if(reduce_dim == 0) {
          out(0, i0, i1) = sum / static_cast<value_type>(reduce_size);
        } else if(reduce_dim == 1) {
          out(i0, 0, i1) = sum / static_cast<value_type>(reduce_size);
        } else {
          out(i0, i1, 0) = sum / static_cast<value_type>(reduce_size);
        }
      })
    );
}

template <class Sender,
          class Scheduler,
          class InoutView,
          class InputView,
          std::enable_if_t<InoutView::rank()==3 && InputView::rank()==3, std::nullptr_t> = nullptr>
stdexec::sender auto axpy_sender(Sender&& sender,
                                 Scheduler&& scheduler,
                                 const InoutView& x,
                                 const InputView& y,
                                 InoutView& z,
                                 const typename InoutView::value_type beta=1,
                                 const typename InoutView::value_type alpha=1) {
  // Outplace: 3D + 1D -> 3D
  assert( x.extents() == z.extents() );
  const auto nx0 = x.extent(0), nx1 = x.extent(1), nx2 = x.extent(2);
  const auto ny0 = y.extent(0), ny1 = y.extent(1), ny2 = y.extent(2);
  const std::size_t n = x.size();
  int broadcast_case = 0;

  if( x.extents() == y.extents() ) {
    broadcast_case = 0;
  } else if( ny0 == 1 && ny0 < nx0 && ny1 == nx1 && ny2 == nx2 ) {
    broadcast_case = 1;
  } else if( ny0 == 1 && ny0 < nx0 && ny1 == 1 && ny1 < nx1 && ny2 == nx2 ) {
    broadcast_case = 2;
  } else if( ny0 == 1 && ny0 < nx0 && ny1 == nx1 && ny2 == 1 && ny2 < nx2 ) {
    broadcast_case = 3;
  } else if( ny0 == nx0 && ny1 == 1 && ny1 < nx1 && ny2 == nx2 ) {
    broadcast_case = 4;
  } else if( ny0 == nx0 && ny1 == 1 && ny1 < nx1 && ny2 == 1 && ny2 < nx2 ) {
    broadcast_case = 5;
  } else if( ny0 == nx0 && ny1 == nx1 && ny2 == 1 && ny2 < nx2 ) {
    broadcast_case = 6;
  } else {
    std::runtime_error("Cannot broadcast y to x. Check the shapes of x and y.");
  }

  return sender |
    exec::on(scheduler, stdexec::bulk(n,
      [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
        const int i0  = idx % nx0;
        const int i12 = idx / nx0;
        const int i1  = i12%nx1;
        const int i2  = i12/nx1;
        int iy0 = i0, iy1 = i1, iy2 = i2;
        if(broadcast_case == 1) {
          iy0 = 0;
        } else if(broadcast_case == 2) {
          iy0 = 0; iy1 = 0;
        } else if(broadcast_case == 3) {
          iy0 = 0; iy2 = 0;
        } else if(broadcast_case == 4) {
          iy1 = 0;
        } else if(broadcast_case == 5) {
          iy1 = 0; iy2 = 0;
        } else if(broadcast_case == 6) {
          iy2 = 0;
        }
        z(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(iy0, iy1, iy2);
      })
    );
}

template <class Sender, class Scheduler, class ViewType>
stdexec::sender auto deep_copy_sender(Sender&& sender, Scheduler&& scheduler, const ViewType& in, ViewType& out) {
  assert(in.extents() == out.extents());
  const auto n = in.size();
  const auto* in_data = in.data_handle();
  auto* out_data = out.data_handle();

  return sender |
    exec::on(scheduler, stdexec::bulk(n,
      [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
        out_data[idx] = in_data[idx];
      })
    );
}

using letkf_config_type = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t, double>;

class LETKFSolver {
//public:
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
  template <class Scheduler>
  stdexec::sender auto solve_axpy_sender(Scheduler&& scheduler) {
    auto X = X_.mdspan();
    auto Y = Y_.mdspan();
    auto dX = dX_.mdspan();
    auto dY = dY_.mdspan();
    auto x_mean = x_mean_.mdspan();
    auto y_mean = y_mean_.mdspan();
    auto I = I_.mdspan();
    auto Q = Q_.mdspan();

    auto _meanX_sender = mean_sender(stdexec::just(), scheduler, X, x_mean, 1);
    auto _subtractXmean_sender = axpy_sender(_meanX_sender, scheduler, X, x_mean, dX, -1);
    auto _meanY_sender = mean_sender(_subtractXmean_sender, scheduler, Y, y_mean, 1);
    auto _subtractYmean_sender = axpy_sender(_meanY_sender, scheduler, Y, y_mean, dY, -1);
    auto _deep_copy_sender = deep_copy_sender(_subtractYmean_sender, scheduler, I, Q);

    return _deep_copy_sender;
  }

  void solve_axpy() {
    auto X = X_.mdspan();
    auto Y = Y_.mdspan();
    auto dX = dX_.mdspan();
    auto dY = dY_.mdspan();
    auto x_mean = x_mean_.mdspan();
    auto y_mean = y_mean_.mdspan();

    // Ensemble average
    Impl::mean(X, x_mean, 1); // (n_stt, n_ens, n_batch) -> (n_stt, 1, n_batch)
    Impl::mean(Y, y_mean, 1); // (n_obs, n_ens, n_batch) -> (n_obs, 1, n_batch)

    // dX = X - mean(X), dY = Y - mean(Y)
    Impl::axpy(X, x_mean, dX, -1); // (n_stt, n_ens, n_batch) - (n_stt, 1, n_batch) -> (n_stt, n_ens, n_batch)
    Impl::axpy(Y, y_mean, dY, -1); // (n_obs, n_ens, n_batch) - (n_obs, 1, n_batch) -> (n_obs, n_ens, n_batch)

    // Q = (Ne-1)I/beta + dY^T * rR * dY
    auto rR = rR_.mdspan(); // (n_obs, n_obs, n_batch)
    auto I = I_.mdspan();
    auto Q = Q_.mdspan();
    auto tmp_oe = tmp_oe_.mdspan();
    const value_type beta = (static_cast<int>(n_ens_) - 1) / beta_;
    Impl::deep_copy(I, Q); // (n_ens, n_ens, n_batch)
  }

  void solve_evd() {
    auto X = X_.mdspan();
    auto dX = dX_.mdspan();
    auto dY = dY_.mdspan();
    auto yo = yo_.mdspan();
    auto x_mean = x_mean_.mdspan();
    auto y_mean = y_mean_.mdspan();

    // dyo = yo - mean(Y)
    Impl::axpy(yo, y_mean, -1); // (n_obs, 1, n_batch)

    // Q = (Ne-1)I/beta + dY^T * rR * dY
    auto rR = rR_.mdspan(); // (n_obs, n_obs, n_batch)
    auto Q = Q_.mdspan();
    auto tmp_oe = tmp_oe_.mdspan();
    const value_type beta = (static_cast<int>(n_ens_) - 1) / beta_;
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
