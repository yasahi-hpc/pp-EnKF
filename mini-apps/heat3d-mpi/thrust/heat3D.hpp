#ifndef __HEAT3D_HPP__
#define __HEAT3D_HPP__

#include <iostream>
#include <thrust/execution_policy.h>
#include "mpi_comm.hpp"
#include "../config.hpp"
#include "../types.hpp"
#include "../timer.hpp"
#include "grid.hpp"
#include "variable.hpp"
#include "functors.hpp"

using counting_iterator = thrust::counting_iterator<size_type>;

template <typename RealType>
void initialize(const Config& conf,
                const Grid<RealType>& grid,
                Comm& comm,
                Variable<RealType>& variables) {

  // Print parallelization
  if(comm.is_master()) {
    auto cart_rank = comm.cart_rank();
    auto topology  = comm.topology();
    std::cout << "Parallelization (px, py, pz) = " << topology.at(0) << ", " << topology.at(1) << ", " << topology.at(2) << std::endl;
    std::cout << "Local (nx, ny, nz) = " << conf.nx_ << ", " << conf.ny_ << ", " << conf.nz_ << std::endl;
    std::cout << "Global (nx, ny, nz) = " << conf.gnx_ << ", " << conf.gny_ << ", " << conf.gnz_ << "\n" << std::endl;
  }

  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const auto x = grid.x();
  const auto y = grid.y();
  const auto z = grid.z();

  auto u = variables.u();

  thrust::for_each(thrust::device,
                   counting_iterator(0), counting_iterator(0)+n,
                   init_functor(conf, x, y, z, u));
}

template <typename RealType>
void solve(const Config& conf,
           Comm& comm,
           Variable<RealType>& variables,
           std::vector<Timer*>& timers) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;

  auto u = variables.u();
  auto un = variables.un();
  auto x_mask = variables.x_mask();
  auto y_mask = variables.y_mask();
  auto z_mask = variables.z_mask();

  for(std::size_t i=0; i<conf.nbiter_; i++) {
    timers[MainLoop]->begin();

    timers[HaloPack]->begin();
    comm.pack(u);
    timers[HaloPack]->end();

    timers[HaloComm]->begin();
    comm.commP2P();
    timers[HaloComm]->end();

    timers[HaloUnpack]->begin();
    comm.unpack(u);
    timers[HaloUnpack]->end();

    timers[Heat]->begin();
    thrust::for_each(thrust::device,
                     counting_iterator(0), counting_iterator(0)+n,
                     heat3d_functor(conf, x_mask, y_mask, z_mask, u, un));
    timers[Heat]->end();

    timers[MainLoop]->end();
  }
}

template <typename RealType>
void finalize(const Config& conf,
              const Grid<RealType>& grid,
              Comm& comm,
              Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const double time = conf.dt_ * conf.nbiter_;

  const auto x = grid.x();
  const auto y = grid.y();
  const auto z = grid.z();

  auto u = variables.u();
  auto un = variables.un();

  thrust::for_each(thrust::device,
                   counting_iterator(0), counting_iterator(0)+n,
                   analytical_solution_functor(conf, time, x, y, z, un));

  // Check errors
  // un: analytical, u: numerical solutions
  auto L2norm = thrust::transform_reduce(
                  thrust::device,
                  counting_iterator(0), counting_iterator(0) + n,
                  [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t idx) {
                    const int h = conf.halo_width_;
                    const std::size_t ix  = idx % conf.nx_ + h;
                    const std::size_t iyz = idx / conf.nx_;
                    const std::size_t iy  = iyz % conf.ny_ + h;
                    const std::size_t iz  = iyz / conf.ny_ + h;

                    auto diff = un(ix, iy, iz) - u(ix, iy, iz);
                    return diff * diff;
                  },
                  0.0,
                  thrust::plus<double>()
                );

  double L2norm_global = 0;
  MPI_Reduce(&L2norm, &L2norm_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(comm.is_master()) {
    std::cout << "L2 norm: " << sqrt(L2norm_global) << std::endl;
  }
}

static void report_performance(const Config& conf, double seconds) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  double GBytes = static_cast<double>(n) * static_cast<double>(conf.nbiter_) * 2 * sizeof(double) / 1.e9;

  // 9 Flop per iteration
  double GFlops = static_cast<double>(n) * static_cast<double>(conf.nbiter_) * 9 / 1.e9;

  #if defined(ENABLE_OPENMP)
    std::cout << "OpenMP backend" << std::endl;
  #else
    std::cout << "CUDA backend" << std::endl;
  #endif

  if(conf.is_async_) {
    std::cout << "Communication and Computation Overlap" << std::endl;
  }
  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth: " << GBytes / seconds << " [GB/s]" << std::endl;
  std::cout << "Flops: " << GFlops / seconds << " [GFlops]" << std::endl;
}

#endif
