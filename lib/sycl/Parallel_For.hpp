#ifndef SYCL_PARALLEL_FOR_HPP
#define SYCL_PARALLEL_FOR_HPP

#include <experimental/mdspan>
#include <sycl/sycl.hpp>
#include "../Iteration.hpp"

namespace stdex = std::experimental;

constexpr std::size_t nb_threads = 256;
constexpr std::size_t nb_threads_x = 32, nb_threads_y = 8, nb_threads_z = 1;

namespace Impl {
  inline auto get_block_size(const std::size_t n, const std::size_t nb_threads) {
    return ( (n + nb_threads - 1) / nb_threads ) * nb_threads;
  }

  // Only for 1D case
  template <class IteratePolicy, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int >, std::nullptr_t> = nullptr>
  void for_each(sycl::queue& q, const IteratePolicy iterate_policy, const FunctorType f) {
    static_assert( IteratePolicy::rank() == 1 );
    const auto start = iterate_policy.start();
    const auto start0 = start[0];
    auto n = iterate_policy.size();

    // Sycl range
    sycl::range<1> global_range {n};

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
        global_range,
        [=](sycl::id<1> idx) {
          const int i0 = idx.get(0) + start0;
          f(i0);
        }
      );
    });
    q.wait();
  }

  template <class IteratePolicy, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int, int >, std::nullptr_t> = nullptr>
  void for_each(sycl::queue& q, const IteratePolicy iterate_policy, const FunctorType f) {
    static_assert( IteratePolicy::rank() == 2 );
    //structured binding cannot be captured
    //auto [start0, start1] = iterate_policy.starts();
    //auto [n0, n1] = iterate_policy.strides();
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1];
    const auto n0 = strides[0], n1 = strides[1];
    auto n = iterate_policy.size();
    
    auto nb_blocks_x = get_block_size(n0, nb_threads_x);
    auto nb_blocks_y = get_block_size(n1, nb_threads_y);

    if(std::is_same_v<typename IteratePolicy::layout_type, stdex::layout_left>) {
      // Get the local range for the 2D parallel for loop
      sycl::range<2> local_range {nb_threads_x, nb_threads_y};

      // Sycl range
      sycl::range<2> global_range {nb_blocks_x, nb_blocks_y};

      // Create a 2D nd_range using the global and local ranges
      sycl::nd_range<2> nd_range(global_range, local_range);

      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
          nd_range,
          [=](sycl::nd_item<2> item) {
            const int i0 = item.get_global_id(0);
            const int i1 = item.get_global_id(1);
            if(i0 < n0 && i1 < n1) {
              f(i0+start0, i1+start1);
            }
          }
        );
      });
      q.wait();
    } else {
      // Get the local range for the 2D parallel for loop
      sycl::range<2> local_range {nb_threads_y, nb_threads_x};

      // Sycl range
      sycl::range<2> global_range {nb_blocks_y, nb_blocks_x};

      // Create a 2D nd_range using the global and local ranges
      sycl::nd_range<2> nd_range(global_range, local_range);

      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
          nd_range,
          [=](sycl::nd_item<2> item) {
            const int i0 = item.get_global_id(1);
            const int i1 = item.get_global_id(0);
            if(i0 < n0 && i1 < n1) {
              f(i0+start0, i1+start1);
            }
          }
        );
      });
      q.wait();
    }
  }

  template <class IteratePolicy, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int >, std::nullptr_t> = nullptr>
  void for_each(sycl::queue& q, const IteratePolicy iterate_policy, const FunctorType f) {
    static_assert( IteratePolicy::rank() == 3 );
    //structured binding cannot be captured
    //auto [start0, start1, start2] = iterate_policy.start();
    //auto [n0, n1, n2] = iterate_policy.strides();
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1], start2 = start[2];
    const auto n0 = strides[0], n1 = strides[1], n2 = strides[2];
    auto n = iterate_policy.size();

    auto nb_blocks_x = get_block_size(n0, nb_threads_x);
    auto nb_blocks_y = get_block_size(n1, nb_threads_y);
    auto nb_blocks_z = get_block_size(n2, nb_threads_z);

    if(std::is_same_v<typename IteratePolicy::layout_type, stdex::layout_left>) {
      // Get the local range for the 3D parallel for loop
      sycl::range<3> local_range {nb_threads_x, nb_threads_y, nb_threads_z};

      // Sycl range
      sycl::range<3> global_range {nb_blocks_x, nb_blocks_y, nb_blocks_z};

      // Create a 3D nd_range using the global and local ranges
      sycl::nd_range<3> nd_range(global_range, local_range);

      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
          nd_range,
          [=](sycl::nd_item<3> item) {
            const int i0 = item.get_global_id(0);
            const int i1 = item.get_global_id(1);
            const int i2 = item.get_global_id(2);
            if(i0 < n0 && i1 < n1 && i2 < n2) {
              f(i0+start0, i1+start1, i2+start2);
            }
          }
        );
      });
      q.wait();
    } else {
      // Get the local range for the 3D parallel for loop
      sycl::range<3> local_range {nb_threads_z, nb_threads_y, nb_threads_x};

      // Sycl range
      sycl::range<3> global_range {nb_blocks_z, nb_blocks_y, nb_blocks_x};

      // Create a 3D nd_range using the global and local ranges
      sycl::nd_range<3> nd_range(global_range, local_range);

      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
          nd_range,
          [=](sycl::nd_item<3> item) {
            const int i0 = item.get_global_id(2);
            const int i1 = item.get_global_id(1);
            const int i2 = item.get_global_id(0);
            if(i0 < n0 && i1 < n1 && i2 < n2) {
              f(i0+start0, i1+start1, i2+start2);
            }
          }
        );
      });
      q.wait();
    }
  }

  template <class IteratePolicy, class FunctorType,
            std::enable_if_t<std::is_invocable_v< FunctorType, int, int, int, int >, std::nullptr_t> = nullptr>
  void for_each(sycl::queue& q, const IteratePolicy iterate_policy, const FunctorType f) {
    static_assert( IteratePolicy::rank() == 4 );
    //structured binding cannot be captured
    //auto [start0, start1, start2, start3] = iterate_policy.starts();
    //auto [n0, n1, n2, n3] = iterate_policy.strides();
    const auto start = iterate_policy.start();
    const auto strides = iterate_policy.strides();
    const auto start0 = start[0], start1 = start[1], start2 = start[2], start3 = start[3];
    const auto n0 = strides[0], n1 = strides[1], n2 = strides[2], n3 = strides[3];
    auto n = iterate_policy.size();

    if(std::is_same_v<typename IteratePolicy::layout_type, stdex::layout_left>) {
      auto nb_blocks_x = get_block_size(n0, nb_threads_x);
      auto nb_blocks_y = get_block_size(n1, nb_threads_y);
      auto nb_blocks_z = get_block_size(n2 * n3, nb_threads_z);

      // Get the local range for the 3D parallel for loop
      sycl::range<3> local_range {nb_threads_x, nb_threads_y, nb_threads_z};

      // Sycl range
      sycl::range<3> global_range {nb_blocks_x, nb_blocks_y, nb_blocks_z};

      // Create a 3D nd_range using the global and local ranges
      sycl::nd_range<3> nd_range(global_range, local_range);

      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
          nd_range,
          [=](sycl::nd_item<3> item) {
            const int i0 = item.get_global_id(0);
            const int i1 = item.get_global_id(1);
            const int i23 = item.get_global_id(2);
            if(i0 < n0 && i1 < n1 && i23 < n2 * n3) {
              const int i2 = i23 % n2 + start2;
              const int i3 = i23 / n2 + start3;
              f(i0+start0, i1+start1, i2, i3);
            }
          }
        );
      });
      q.wait();
    } else {
      auto nb_blocks_x = get_block_size(n3, nb_threads_x);
      auto nb_blocks_y = get_block_size(n2, nb_threads_y);
      auto nb_blocks_z = get_block_size(n1 * n0, nb_threads_z);

      // Get the local range for the 3D parallel for loop
      sycl::range<3> local_range {nb_threads_x, nb_threads_y, nb_threads_z};

      // Sycl range
      sycl::range<3> global_range {nb_blocks_x, nb_blocks_y, nb_blocks_z};

      // Create a 3D nd_range using the global and local ranges
      sycl::nd_range<3> nd_range(global_range, local_range);

      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
          nd_range,
          [=](sycl::nd_item<3> item) {
            const int i01 = item.get_global_id(2);
            const int i0 = i01 / n1;
            const int i1 = i01 % n1;
            const int i2 = item.get_global_id(1);
            const int i3 = item.get_global_id(0);
            if(i01 < n0 * n1 && i2 < n2 && i3 < n3) {
              f(i0+start0, i1+start1, i2+start2, i3+start3);
            }
          }
        );
      });
      q.wait();
    }
  }
};

#endif
