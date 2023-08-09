#ifndef __HIP_RANDOM_HPP__
#define __HIP_RANDOM_HPP__

#include <rocrand/rocrand.hpp>
#include <type_traits>
#include "HIP_Helper.hpp"

namespace Impl {
  template <typename RealType,
            typename std::enable_if<std::is_same<RealType, float>::value ||
                                    std::is_same<RealType, double>::value 
                                   >::type * = nullptr>
  struct Random {
  private:
    rocrand_generator generator_;

  public:
    Random() {
      rocrand_rng_type rng_type = ROCRAND_RNG_PSEUDO_DEFAULT;
      rocrand_create_generator(&generator_, rng_type);
      rocrand_initialize_generator(generator_);
      rocrand_set_seed(generator_, 1991ul);
    }

    ~Random() {
      rocrand_destroy_generator(generator_);
    }

    void normal(RealType* outputPtr, std::size_t n, RealType mean, RealType stddev) {
      normal_(outputPtr, n, mean, stddev);
      SafeHIPCall( hipDeviceSynchronize() );
    }

  private:
    void normal_(float* outputPtr, std::size_t n, float mean, float stddev) {
      rocrand_generate_normal(generator_, outputPtr, n, mean, stddev); 
    }

    void normal_(double* outputPtr, std::size_t n, double mean, double stddev) {
      rocrand_generate_normal_double(generator_, outputPtr, n, mean, stddev); 
    }
  };
};

#endif
