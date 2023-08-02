#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexec/execution.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <exec/static_thread_pool.hpp>
#include "nvexec/stream_context.cuh"
#include <exec/on.hpp>
#include "stream.hpp"

using counting_iterator = thrust::counting_iterator<std::size_t>;

#if defined(ENABLE_OPENMP)
  using Vector = thrust::host_vector<double>;
#else
  using Vector = thrust::device_vector<double>;
#endif

constexpr std::size_t ARRAY_SIZE = 128*128*128*128;
constexpr std::size_t nbiter = 100;
constexpr double start_A = 0.1;
constexpr double start_B = 0.2;
constexpr double start_C = 0.0;
constexpr double start_Scalar = 0.4;
std::string csv_separator = ",";

template <class Sender>
void exec_with_timer(Sender&& sender, std::vector<double>& timing) {
  auto start = std::chrono::high_resolution_clock::now();
  stdexec::sync_wait(sender);
  auto end = std::chrono::high_resolution_clock::now();
  timing.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count());
}

template <class Iterator>
struct simple_range {
  Iterator first;
  Iterator last;
};

template <class Iterator>
auto begin(simple_range<Iterator>& rng) {
  return rng.first;
}

template <class Iterator>
auto end(simple_range<Iterator>& rng) {
  return rng.last;
}

template <class OutputType, class UnarayOperation>
void average(const std::size_t n, UnarayOperation const unary_op, OutputType &result) {
  auto init = result;
  result = thrust::transform_reduce(thrust::device,
                                    counting_iterator(0), counting_iterator(0) + n,
                                    [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t i) {
                                      return unary_op(i);
                                    },
                                    init,
                                    thrust::plus<double>()
                                   );
  result /= n;
}

template <typename RealType>
void checkSolution(const std::size_t nbiter,
                   const Vector& a,
                   const Vector& b,
                   const Vector& c,
                   const RealType& sum) {
  // Generate correct solution
  RealType gold_A = start_A;
  RealType gold_B = start_B;
  RealType gold_C = start_C;
  RealType gold_sum = 0;

  for(std::size_t iter = 0; iter < nbiter; iter++) {
    gold_C = gold_A; // copy
    gold_B = start_Scalar * gold_C; // mult
    gold_C = gold_A + gold_B; // add

    gold_A = gold_B + start_Scalar * gold_C;
  }

  gold_sum = gold_A * gold_B * ARRAY_SIZE;

  // Calculate the average error
  RealType* ptr_a = (RealType *)thrust::raw_pointer_cast(a.data());
  RealType* ptr_b = (RealType *)thrust::raw_pointer_cast(b.data());
  RealType* ptr_c = (RealType *)thrust::raw_pointer_cast(c.data());
  RealType err_A = 0, err_B = 0, err_C = 0;
  average(ARRAY_SIZE, 
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t i){
            return fabs(ptr_a[i] - gold_A);
          }, err_A);

  average(ARRAY_SIZE, 
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t i){
            return fabs(ptr_b[i] - gold_B);
          }, err_B);

  average(ARRAY_SIZE, 
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t i){
            return fabs(ptr_c[i] - gold_C);
          }, err_C);

  double err_sum = fabs(sum - gold_sum);

  double epsi = std::numeric_limits<RealType>::epsilon() * 100.0;
  if(err_A > epsi)
    std::cerr
      << "Validation failed on a[]. Average error " << err_A
      << std::endl;

  if(err_B > epsi)
    std::cerr
      << "Validation failed on b[]. Average error " << err_B
      << std::endl;

  if(err_C > epsi)
    std::cerr
      << "Validation failed on c[]. Average error " << err_C
      << std::endl;

  // Check sum to 8 decimal places
  if(err_sum > 1.e-8)
    std::cerr
      << "Validation failed on sum. Error " << err_sum
      << std::endl
      << "Sum was " << std::setprecision(15) << sum << " but should be " << gold_sum
      << std::endl;
};

int main(int argc, char *argv[]) {
  #if defined(ENABLE_OPENMP)
    exec::static_thread_pool pool{std::thread::hardware_concurrency()};
    auto scheduler = pool.get_scheduler();
  #else
    // Declare a CUDA stream
    nvexec::stream_context stream_ctx{};
    auto scheduler = stream_ctx.get_scheduler();
  #endif

  // Declare device vectors
  Vector a(ARRAY_SIZE);
  Vector b(ARRAY_SIZE);
  Vector c(ARRAY_SIZE);
  double* ptr_a = (double *)thrust::raw_pointer_cast(a.data());
  double* ptr_b = (double *)thrust::raw_pointer_cast(b.data());
  double sum = 0.0;

  auto init = stdexec::just()
            | exec::on( scheduler, stdexec::bulk( ARRAY_SIZE, init_functor(start_A, start_B, start_C, a, b, c) ) );
  stdexec::sync_wait(std::move(init));

  // List of times
  std::vector<std::vector<double>> timings(5);

  for (std::size_t iter = 0; iter < nbiter; iter++) {
    auto copy = stdexec::just()
              | exec::on( scheduler, stdexec::bulk( ARRAY_SIZE, copy_functor(a, c) ) );

    auto mul = stdexec::just()
             | exec::on( scheduler, stdexec::bulk( ARRAY_SIZE, mul_functor(start_Scalar, b, c) ) );

    auto add = stdexec::just()
             | exec::on( scheduler, stdexec::bulk( ARRAY_SIZE, add_functor(a, b, c) ) );

    auto triad = stdexec::just()
               | exec::on( scheduler, stdexec::bulk( ARRAY_SIZE, triad_functor(start_Scalar, a, b, c) ) );

    /*
    auto dot = stdexec::just()
             | exec::on( scheduler, stdexec::bulk( ARRAY_SIZE, dot_functor(buffer, a, b) ) )
             | stdexec::let_value(
                [=](){
                  return stdexec::just( simple_range<double*>{ptr_buffer, ptr_buffer+ARRAY_SIZE} );
                })
             | nvexec::reduce();
     */

    exec_with_timer(std::move(copy),  timings[0]);
    exec_with_timer(std::move(mul),   timings[1]);
    exec_with_timer(std::move(add),   timings[2]);
    exec_with_timer(std::move(triad), timings[3]);

    auto start = std::chrono::high_resolution_clock::now();
    sum = thrust::transform_reduce(thrust::device,
                                   counting_iterator(0), counting_iterator(0) + ARRAY_SIZE,
                                   [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t i) {
                                     return ptr_a[i] * ptr_b[i];
                                   },
                                   0.0,
                                   thrust::plus<double>()
                                  );
    auto end = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count());
    
    /*
    auto start = std::chrono::high_resolution_clock::now();
    auto [sum_tmp] = stdexec::sync_wait(std::move(dot)).value();
    auto end = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count());
    sum = sum_tmp;
    */
  }

  checkSolution(nbiter, a, b, c, sum);

  #if defined(ENABLE_OPENMP)
    std::cout << "OpenMP backend" << std::endl;
  #else
    std::cout << "CUDA backend" << std::endl;
  #endif

  std::cout
    << "function" << csv_separator
    << "num_times" << csv_separator
    << "n_elements" << csv_separator
    << "sizeof" << csv_separator
    << "max_gbytes_per_sec" << csv_separator
    << "min_runtime" << csv_separator
    << "max_runtime" << csv_separator
    << "avg_runtime" << std::endl;

  std::string labels[5] = {"Copy", "Mul", "Add", "Triad", "Dot"};
  size_t sizes[5] = {
    2 * sizeof(double) * ARRAY_SIZE,
    2 * sizeof(double) * ARRAY_SIZE,
    3 * sizeof(double) * ARRAY_SIZE,
    3 * sizeof(double) * ARRAY_SIZE,
    2 * sizeof(double) * ARRAY_SIZE
  };

  for (int i = 0; i < 5; i++) {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(nbiter - 1);

    // Display results
    std::cout
      << labels[i] << csv_separator
      << nbiter << csv_separator
      << ARRAY_SIZE << csv_separator
      << sizeof(double) << csv_separator
      << 1.0E-9 * sizes[i] / (*minmax.first) << csv_separator
      << *minmax.first << csv_separator
      << *minmax.second << csv_separator
      << average
      << std::endl;
  }

  return 0;
}
