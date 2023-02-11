#ifndef __HEAT3D_HPP__
#define __HEAT3D_HPP__

#include <iostream>
#include <ranges>
#include <execution>
#include "mpi_comm.hpp"
#include "../config.hpp"
#include "../types.hpp"
#include "grid.hpp"
#include "variable.hpp"
#include "functors.hpp"

template <typename RealType>
void testComm(const Config& conf,
              Comm& comm,
              Variable<RealType>& variables) {
  auto cart_rank = comm.cart_rank();
  auto topology  = comm.topology();

  auto u = variables.u();
  auto un = variables.un();

  // fill u and un
  const int h = conf.halo_width_;
  for(std::size_t iz=h; iz<conf.nz_+h; iz++) {
    for(std::size_t iy=h; iy<conf.ny_+h; iy++) {
      for(std::size_t ix=h; ix<conf.nx_+h; ix++) {
        std::size_t gix = ix + conf.nx_ * cart_rank.at(0);
        std::size_t giy = iy + conf.ny_ * cart_rank.at(1);
        std::size_t giz = iz + conf.nz_ * cart_rank.at(2);

        u(ix, iy, iz)  = ((double)giz * conf.gny_ + (double)giy) * conf.gnx_ + gix;
        un(ix, iy, iz) = ((double)giz * conf.gny_ + (double)giy) * conf.gnx_ + gix;
      }
    }
  }

  comm.exchangeHalos(u);

  auto print_error = [&](std::size_t ix, std::size_t iy, std::size_t iz,
                         std::size_t gix, std::size_t giy, std::size_t giz) {
    auto diff = un(ix, iy, iz) - u(ix, iy, iz);
    if (fabs(diff) > .1) {
      printf("Pb at rank %d (%d, %d, %d) u(%zu, %zu, %zu): %lf, un(%zu, %zu, %zu): %lf, error: %lf\n",
             comm.rank(), cart_rank.at(0), cart_rank.at(1), cart_rank.at(2), ix, iy, iz, u(ix, iy, iz), gix, giy, giz, un(ix, iy, iz), diff);
    }
  };

  // Fill halos manually
  for(int iz=h; iz<conf.nz_+h; iz++) {
    for(int iy=h; iy<conf.ny_+h; iy++) {
      int gix_left  = 1 + conf.nx_ * ( ( cart_rank.at(0) + topology.at(0) + 1) % topology.at(0) );
      int gix_right = conf.nx_ + conf.nx_ * ( ( cart_rank.at(0) + topology.at(0) - 1 ) % topology.at(0) );
      int giy = iy + conf.ny_ * cart_rank.at(1);
      int giz = iz + conf.nz_ * cart_rank.at(2);

      un(0, iy, iz)          = ((double)giz * conf.gny_ + (double)giy) * conf.gnx_ + gix_right;
      un(conf.nx_+1, iy, iz) = ((double)giz * conf.gny_ + (double)giy) * conf.gnx_ + gix_left;

      print_error(0, iy, iz, gix_right, giy, giz);
      print_error(conf.nx_+1, iy, iz, gix_left, giy, giz);
    }
  }

  for(int iz=h; iz<conf.nz_+h; iz++) {
    for(int ix=h; ix<conf.nx_+h; ix++) {
      int giy_left  = 1 + conf.ny_ * ( ( cart_rank.at(1) + topology.at(1) + 1 ) % topology.at(1) );
      int giy_right = conf.ny_ + conf.ny_ * ( ( cart_rank.at(1) + topology.at(1) - 1 ) % topology.at(1) );
      int gix = ix + conf.nx_ * cart_rank.at(0);
      int giz = iz + conf.nz_ * cart_rank.at(2);

      un(ix, 0, iz)          = ((double)giz * conf.gny_ + (double)giy_right) * conf.gnx_ + gix;
      un(ix, conf.ny_+1, iz) = ((double)giz * conf.gny_ + (double)giy_left) * conf.gnx_ + gix;
      print_error(ix, 0, iz, gix, giy_right, giz);
      print_error(ix, conf.ny_+1, iz, gix, giy_left, giz);
    }
  }

  for(int iy=h; iy<conf.ny_+h; iy++) {
    for(int ix=h; ix<conf.nx_+h; ix++) {
      int giz_left  = 1 + conf.nz_ * ( ( cart_rank.at(2) + topology.at(2) + 1 ) % topology.at(2) );
      int giz_right = conf.nz_ + conf.nz_ * ( ( cart_rank.at(2) + topology.at(2) - 1 ) % topology.at(2) );
      int gix = ix + conf.nx_ * cart_rank.at(0);
      int giy = iy + conf.ny_ * cart_rank.at(1);

      un(ix, iy, 0)          = ((double)giz_right * conf.gny_ + (double)giy) * conf.gnx_ + gix;
      un(ix, iy, conf.nz_+1) = ((double)giz_left  * conf.gny_ + (double)giy) * conf.gnx_ + gix;
      print_error(ix, iy, 0, gix, giy, giz_right);
      print_error(ix, iy, conf.nz_+1, gix, giy, giz_left);
    }
  }

}

template <typename RealType>
void initialize(const Config& conf,
                Grid<RealType>& grid,
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

  // Test for communication
  testComm(conf, comm, variables);

  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const auto x = grid.x();
  const auto y = grid.y();
  const auto z = grid.z();

  auto u = variables.u();

  std::for_each_n(std::execution::par_unseq,
                  std::views::iota(0).begin(), n,
                  init_functor(conf, x, y, z, u));
}

template <typename RealType>
void solve(const Config& conf,
           Comm& comm,
           Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;

  auto u = variables.u();
  auto un = variables.un();
  const auto x_mask = variables.x_mask();
  const auto y_mask = variables.y_mask();
  const auto z_mask = variables.z_mask();

  for(std::size_t i=0; i<conf.nbiter_; i++) {
    comm.exchangeHalos(u);
    std::for_each_n(std::execution::par_unseq,
                    std::views::iota(0).begin(), n,
                    heat3d_functor(conf, x_mask, y_mask, z_mask, u, un));
    std::swap(u, un);
  }
}

template <typename RealType>
void finalize(const Config& conf,
              Grid<RealType>& grid,
              Comm& comm,
              Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const double time = conf.dt_ * conf.nbiter_;

  const auto x = grid.x();
  const auto y = grid.y();
  const auto z = grid.z();

  auto u = variables.u();
  auto un = variables.un();

  std::for_each_n(std::execution::par_unseq,
                  std::views::iota(0).begin(), n,
                  analytical_solution_functor(conf, time, x, y, z, un));

  // Check errors
  // un: analytical, u: numerical solutions
  auto L2norm = std::transform_reduce(std::execution::par_unseq,
                                      std::views::iota(0).begin(), std::views::iota(0).begin() + n,
                                      0.0,
                                      std::plus<double>(),
                                      [=] (const std::size_t idx) {
                                        const int h = conf.halo_width_;
                                        const std::size_t ix  = idx % conf.nx_ + h;
                                        const std::size_t iyz = idx / conf.nx_;
                                        const std::size_t iy  = iyz % conf.ny_ + h;
                                        const std::size_t iz  = iyz / conf.ny_ + h;

                                        auto diff = un(ix, iy, iz) - u(ix, iy, iz);
                                        return diff * diff;
                                      }
                                     );

  double L2norm_global = 0;
  MPI_Reduce(&L2norm, &L2norm_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(comm.is_master()) {
    std::cout << "L2 norm: " << sqrt(L2norm_global) << std::endl;
  }
}

static void report_performance(const Config& conf, double seconds) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const double GBytes = static_cast<double>(n) * static_cast<double>(conf.nbiter_) * 2 * sizeof(double) / 1.e9;

  // 9 Flop per iteration
  const double GFlops = static_cast<double>(n) * static_cast<double>(conf.nbiter_) * 9 / 1.e9;

  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth: " << GBytes / seconds << " [GB/s]" << std::endl;
  std::cout << "Flops: " << GFlops / seconds << " [GFlops]" << std::endl;
}

#endif
