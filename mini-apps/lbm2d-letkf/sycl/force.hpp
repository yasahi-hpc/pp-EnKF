#ifndef FORCE_HPP
#define FORCE_HPP

#include <vector>
#include <random>
#include <sycl/sycl.hpp>
#include <sycl/Parallel_For.hpp>
#include <Random.hpp>
#include "../config.hpp"
#include "types.hpp"

struct Force {
private:
  using value_type = RealView2D::value_type;
  sycl::queue queue_;
  Config conf_;
  Impl::Random<value_type> rand_;
  RealView1D x_, y_;
  RealView1D kx_, ky_;
  RealView1D amp_;
  RealView3D rand_pool_;
  RealView2D fx_, fy_;
  std::size_t n_rand_buf_;
  std::size_t i_rand_outdated_;
  const value_type mean_ = 0.0;
  const value_type stddev_ = 0.84089641525;

public:
  Force()=delete;
  Force(sycl::queue& queue, Config& conf) : queue_(queue), conf_(conf) {
    init_forces(conf);
    update_forces();
  }

  ~Force() {} 

  // Getters
  auto& fx() { return fx_; }
  auto& fy() { return fy_; }

  void update_forces() {
    // forces watanabe97
    i_rand_outdated_ += amp_.size() * 4;

    if( i_rand_outdated_ + amp_.size() * 4 >= n_rand_buf_ ) {
      rand_.normal(rand_pool_.data(), n_rand_buf_, mean_, stddev_);
      i_rand_outdated_ = 0;
    }
    auto [nx, ny] = conf_.settings_.n_;
    const int shift = i_rand_outdated_ / (4 * amp_.size());
    const auto force_n = amp_.size();
    const auto kx = kx_.mdspan();
    const auto ky = ky_.mdspan();
    const auto amp = amp_.mdspan();
    const auto x = x_.mdspan();
    const auto y = y_.mdspan();
    const auto rand_pool = rand_pool_.mdspan();
    //const auto sub_rand_pool = stdex::submdspan(rand_pool, std::full_extent_t, std::full_extent_t, shift);
    auto fx = fx_.mdspan();
    auto fy = fy_.mdspan();

    auto force_lambda = [=](const int ix, const int iy) {
      const auto x_tmp = x(ix);
      const auto y_tmp = y(iy);
      value_type fx_tmp = 0.0, fy_tmp = 0.0;
      for(std::size_t n=0; n<force_n; n++) {
        const auto kx_tmp = kx(n);
        const auto ky_tmp = ky(n);
        const auto theta = 2*M_PI*(kx_tmp*x_tmp + ky_tmp*y_tmp);
        const auto sine = sin(theta);
        const auto cosi = cos(theta);
        const value_type r[4] = {
          rand_pool(n, 0, shift),
          rand_pool(n, 1, shift),
          rand_pool(n, 2, shift),
          rand_pool(n, 3, shift),
        };

        const auto amp_tmp = amp(n);
        fx_tmp += amp_tmp * 2 * M_PI * ky_tmp    * ((r[0]*kx_tmp + r[1]*ky_tmp)*sine + (r[2]*kx_tmp + r[3]*ky_tmp)*cosi);
        fy_tmp += amp_tmp * 2 * M_PI * (-kx_tmp) * ((r[0]*kx_tmp + r[1]*ky_tmp)*sine + (r[2]*kx_tmp + r[3]*ky_tmp)*cosi);
      }
      constexpr value_type R = 0.0;
      fx(ix, iy) = R * fx(ix, iy) + sqrt(1-R*R) * fx_tmp;
      fy(ix, iy) = R * fy(ix, iy) + sqrt(1-R*R) * fy_tmp;
    };

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(queue_, policy2d, force_lambda);
  }

private:
  void init_forces(Config& conf) {
    // initilize spectrum of force
    constexpr int max_search_nmode = 10000;
    const value_type kf = static_cast<value_type>(conf.phys_.kf_);
    const value_type amp_scale = static_cast<value_type>(conf.phys_.fkf_);
    constexpr value_type sigma = 1; // gaussian stdev
    constexpr value_type dk = 2; // clipping width
    const int kmax_search = int(kf+dk);
    long double asum = 0;
    std::vector<value_type> force_kx;
    std::vector<value_type> force_ky;
    std::vector<value_type> force_amp;
    for(int ky=0; ky<kmax_search; ky++) { for(int kx=0; kx<kmax_search; kx++) {
      if(kx == 0 and ky == 0) { continue; }
      const auto k = std::sqrt(ky*ky + kx*kx);
      if(kf-dk <= k and k <= kf+dk) {  /// maltrud91
        const value_type amp = std::exp( - (k-kf)*(k-kf) / sigma ); /// Gaussian decay by |k-kf|
        asum += amp;
        force_kx.push_back(kx);
        force_ky.push_back(ky);
        force_amp.push_back(amp);
      }
      if(force_kx.size() >= max_search_nmode) { break; }
    }}

    // prune modes of force
    constexpr size_t max_nmode = 100;
    if(force_kx.size() > max_nmode) {
      auto tmp_kx = force_kx;
      auto tmp_ky = force_ky;
      auto tmp_am = force_amp;
      std::vector<int> ps(max_nmode);
      std::mt19937 engine(force_kx.size());
      for(auto& p: ps) {
        std::uniform_int_distribution<int> dist(0, max_nmode);
        p = dist(engine);
      }

      force_kx.clear();
      force_ky.clear();
      force_amp.clear();
      for(auto p: ps) {
        force_kx.push_back(tmp_kx.at(p));
        force_ky.push_back(tmp_ky.at(p));
        force_amp.push_back(tmp_am.at(p));
      }
    }

    if(conf_.settings_.ensemble_idx_ == 0) {
      std::cout << "force: " << "n=" << int(force_kx.size()) << ", sum=" << asum << std::endl;
    }

    constexpr value_type config_t0__ = 1.0;
    const value_type force_total_amp = amp_scale * static_cast<value_type>(conf.phys_.u_ref_) / config_t0__;
    for(value_type& amp: force_amp) {
      amp *= force_total_amp / asum;
    }

    n_rand_buf_ = force_amp.size() * 4096;
    i_rand_outdated_ = n_rand_buf_;

    // Define mdspans
    auto [nx, ny] = conf.settings_.n_;
    kx_  = RealView1D(queue_, "kx", force_kx.size());
    ky_  = RealView1D(queue_, "ky", force_ky.size());
    amp_ = RealView1D(queue_, "amp", force_amp.size());
    x_   = RealView1D(queue_, "x", nx);
    y_   = RealView1D(queue_, "y", ny);
    rand_pool_ = RealView3D(queue_, "rand", force_kx.size(), 4, 4096);

    fx_ = RealView2D(queue_, "fx", nx, ny);
    fy_ = RealView2D(queue_, "fy", nx, ny);

    for(std::size_t i=0; i<nx; i++) {
      value_type tmp_x = ( static_cast<value_type>(i) - static_cast<value_type>(nx/2) ) / static_cast<value_type>(nx);
      x_(i) = tmp_x;
    }

    for(std::size_t i=0; i<ny; i++) {
      value_type tmp_y = ( static_cast<value_type>(i) - static_cast<value_type>(ny/2) ) / static_cast<value_type>(ny);
      y_(i) = tmp_y;
    }

    for(std::size_t i=0; i<force_kx.size(); i++) {
      kx_(i) = force_kx.at(i);
    }

    for(std::size_t i=0; i<force_ky.size(); i++) {
      ky_(i) = force_ky.at(i);
    }

    for(std::size_t i=0; i<force_amp.size(); i++) {
      amp_(i) = force_amp.at(i);
    }

  }
};

#endif
