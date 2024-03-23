#ifndef LETKF_HPP
#define LETKF_HPP

#include <sycl/Parallel_For.hpp>
#include <sycl/Transpose.hpp>
#include <utils/mpi_utils.hpp>
#include "letkf_solver.hpp"
#include "../functors.hpp"
#include "../da_functors.hpp"
#include "da_models.hpp"

class LETKF : public DA_Model {
private:
  using value_type = RealView2D::value_type;
  MPIConfig mpi_conf_;

  Impl::blasHandle_t blas_handle_;
  std::unique_ptr<LETKFSolver> letkf_solver_;
  
  /* Views before transpose */ 
  RealView3D xk_; // (n_stt, n_batch, n_ens) = (n_stt, nx*ny)
  RealView3D xk_buffer_; // (n_stt, n_batch, n_ens)
  RealView3D yk_; // (n_obs, n_batch, n_ens) = (n_obs, nx*ny)
  RealView3D yk_buffer_; // (n_obs, n_batch, n_ens)

  value_type d_local_;
  int obs_offset_ = 0;
  int n_obs_local_;
  int n_obs_x_;
  int n_obs_;

public:
  LETKF(sycl::queue& queue, Config& conf, IOConfig& io_conf)=delete;
  LETKF(sycl::queue& queue, Config& conf, IOConfig& io_conf, MPIConfig& mpi_conf) : DA_Model(queue, conf, io_conf), mpi_conf_(mpi_conf) {}

  virtual ~LETKF(){ blas_handle_.destroy(); }
  void initialize() {
    setFileInfo();

    auto [nx, ny] = conf_.settings_.n_;
    const int n_batch0 = nx * ny;
    const int n_stt = conf_.phys_.Q_; // lbm
    constexpr int no_per_grid = 3; // rho, u, v

    const int rloc_len = conf_.settings_.rloc_len_;
    const int obs_interval = conf_.settings_.obs_interval_;
    d_local_ = static_cast<value_type>(rloc_len) * static_cast<value_type>(obs_interval);
    n_obs_local_ = rloc_len * 2;
    n_obs_x_ = n_obs_local_ * 2 + 1;
    const int n_obs = n_obs_x_ * n_obs_x_ * no_per_grid;

    const auto n_ens = mpi_conf_.size();
    const auto n_batch = n_batch0 / mpi_conf_.size();

    xk_ = RealView3D(queue_, "xk", n_stt, n_batch, n_ens);
    yk_ = RealView3D(queue_, "yk", n_obs, n_batch, n_ens);
    xk_buffer_ = RealView3D(queue_, "xk_buffer", n_stt, n_batch, n_ens);
    yk_buffer_ = RealView3D(queue_, "yk_buffer", n_obs, n_batch, n_ens);

    const auto beta = conf_.settings_.beta_;
    letkf_config_type letkf_config = {n_ens, n_stt, n_obs, n_batch, beta};
    letkf_solver_ = std::move( std::unique_ptr<LETKFSolver>(new LETKFSolver(queue_, letkf_config)) ); 
    
    auto rR = letkf_solver_->rR().mdspan();
    const int ny_local = ny/mpi_conf_.size();
    const int y_offset = ny_local * mpi_conf_.rank();

    Iterate_policy<3> policy3d({0, 0, 0}, {n_obs_x_, n_obs_x_, n_batch});
    Impl::for_each(queue_, policy3d, initialize_rR_functor(conf_, y_offset, rR));

    blas_handle_.create();
    blas_handle_.set_stream(queue_);
  }

  void apply(std::unique_ptr<DataVars>& data_vars, const int it, std::vector<Timer*>& timers){
    if(it == 0 || it % conf_.settings_.da_interval_ != 0) return;
    if(mpi_conf_.is_master()) {
      std::cout << __PRETTY_FUNCTION__ << ": t=" << it << std::endl;
    }

    timers[TimerEnum::DA]->begin();
    timers[DA_Load]->begin();
    if(mpi_conf_.is_master()) {
      load(data_vars, it);
    }
    timers[DA_Load]->end();
    setXandY(data_vars, timers);

    timers[DA_LETKF]->begin();
    letkf_solver_->solve();
    timers[DA_LETKF]->end();

    timers[DA_Update]->begin();
    update(data_vars);
    timers[DA_Update]->end();
    timers[TimerEnum::DA]->end();
  }

  void diag(){}
  void finalize(){}

private:
  void setXandY(std::unique_ptr<DataVars>& data_vars, std::vector<Timer*>& timers) {
    /* Set X, Y and yo in letkf solver */

    // set X
    const auto f = data_vars->f().mdspan();
    auto xk = xk_.mdspan();
    auto xk_buffer = xk_buffer_.mdspan();
    auto X = letkf_solver_->X().mdspan();

    timers[DA_Pack_X]->begin();
    Impl::transpose(blas_handle_, f, xk, {2, 0, 1}); // (nx, ny, Q) -> (Q, nx*ny)
    blas_handle_.wait();
    timers[DA_Pack_X]->end();

    timers[DA_All2All_X]->begin();
    all2all(xk, xk_buffer); // xk(n_stt, n_batch, n_ens) -> xk_buffer(n_stt, n_batch, n_ens)
    timers[DA_All2All_X]->end();

    timers[DA_Unpack_X]->begin();
    Impl::transpose(blas_handle_, xk_buffer, X, {0, 2, 1});
    blas_handle_.wait();
    timers[DA_Unpack_X]->end();

    // set Y
    auto yk = yk_.mdspan();
    auto yk_buffer = yk_buffer_.mdspan();
    auto Y = letkf_solver_->Y().mdspan();

    auto [nx, ny] = conf_.settings_.n_;
    auto rho = data_vars->rho().mdspan();
    auto u   = data_vars->u().mdspan();
    auto v   = data_vars->v().mdspan();

    const int y_offset0 = 0;
    auto _yk = Impl::reshape(yk, std::array<std::size_t, 3>({n_obs_x_*n_obs_x_, 3, nx*ny}));
    Iterate_policy<4> yk_pack_policy4d({0, 0, 0, 0}, {n_obs_x_, n_obs_x_, nx, ny});
    timers[DA_Pack_Y]->begin();
    Impl::for_each(queue_, yk_pack_policy4d, pack_y_functor(conf_, y_offset0, rho, u, v, _yk));
    timers[DA_Pack_Y]->end();

    timers[DA_All2All_Y]->begin();
    all2all(yk, yk_buffer); // yk(n_obs, n_batch, n_ens) -> yk_buffer(n_obs, n_batch, n_ens)
    timers[DA_All2All_Y]->end();

    timers[DA_Unpack_Y]->begin();
    Impl::transpose(blas_handle_, yk_buffer, Y, {0, 2, 1}); // (n_obs, n_batch, n_ens) -> (n_obs, n_ens, n_batch)
    blas_handle_.wait();
    timers[DA_Unpack_Y]->end();

    // set yo
    auto rho_obs = data_vars->rho_obs().mdspan();
    auto u_obs   = data_vars->u_obs().mdspan();
    auto v_obs   = data_vars->v_obs().mdspan();
    auto y_obs   = letkf_solver_->y_obs().mdspan();
    timers[DA_Broadcast]->begin();
    broadcast(rho_obs);
    broadcast(u_obs);
    broadcast(v_obs);
    timers[DA_Broadcast]->end();

    const int ny_local = ny/mpi_conf_.size();
    const int y_offset = ny_local * mpi_conf_.rank();
    auto _y_obs = Impl::reshape(y_obs, std::array<std::size_t, 3>({n_obs_x_*n_obs_x_, 3, nx*ny_local}));
    Iterate_policy<4> yo_pack_policy4d({0, 0, 0, 0}, {n_obs_x_, n_obs_x_, nx, ny_local});

    timers[DA_Pack_Obs]->begin();
    Impl::for_each(queue_, yo_pack_policy4d, pack_y_functor(conf_, y_offset, rho_obs, u_obs, v_obs, _y_obs));
    timers[DA_Pack_Obs]->end();
  }

  void update(std::unique_ptr<DataVars>& data_vars) {
    auto X = letkf_solver_->X().mdspan();
    auto xk = xk_.mdspan();
    auto xk_buffer = xk_buffer_.mdspan();
    auto f = data_vars->f().mdspan();
    Impl::transpose(blas_handle_, X, xk_buffer, {0, 2, 1}); // X (n_stt, n_ens, n_batch) -> xk_buffer (n_stt, n_batch, n_ens)
    blas_handle_.wait();
    all2all(xk_buffer, xk); // xk_buffer (n_stt, n_batch, n_ens) -> xk(n_stt, n_batch, n_ens)
    Impl::transpose(blas_handle_, xk, f, {1, 2, 0}); // (Q, nx*ny) -> (nx, ny, Q)
    blas_handle_.wait();

    auto [nx, ny] = conf_.settings_.n_;
    auto rho = data_vars->rho().mdspan();
    auto u   = data_vars->u().mdspan();
    auto v   = data_vars->v().mdspan();

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(queue_, policy2d, macroscopic_functor(conf_, f, rho, u, v));
  }

  template <class ViewType,
            std::enable_if_t<ViewType::rank()==3, std::nullptr_t> = nullptr>
  void all2all(const ViewType& a, ViewType& b) {
    assert( a.extents() == b.extents() );
    MPI_Datatype mpi_datatype = Impl::getMPIDataType<ViewType::value_type>();

    const std::size_t size = a.extent(0) * a.extent(1);
    MPI_Alltoall(a.data_handle(),
                 size,
                 mpi_datatype,
                 b.data_handle(),
                 size,
                 mpi_datatype,
                 mpi_conf_.comm());
  }

  template <class ViewType>
  void broadcast(ViewType& a) {
    MPI_Datatype mpi_datatype = Impl::getMPIDataType<ViewType::value_type>();

    const std::size_t size = a.size();
    MPI_Bcast(a.data_handle(),
              size,
              mpi_datatype,
              0,
              mpi_conf_.comm());
  }
};

#endif