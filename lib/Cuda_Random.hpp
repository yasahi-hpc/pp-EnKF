#ifndef __CUDA_RANDOM_HPP__
#define __CUDA_RANDOM_HPP__

#include <curand.h>
#include <type_traits>
#include "Cuda_Helper.hpp"

namespace Impl {
  template <typename RealType,
            typename std::enable_if<std::is_same<RealType, float>::value ||
                                    std::is_same<RealType, double>::value 
                                   >::type * = nullptr>
  struct Random {
  private:
    curandGenerator_t randgen_;

  public:
    Random() {
      curandRngType_t rng_type = CURAND_RNG_PSEUDO_DEFAULT;
      //curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;
      curandCreateGenerator(&randgen_, rng_type);
      curandSetPseudoRandomGeneratorSeed(randgen_, 1991ul);
    }

    ~Random() {
      curandDestroyGenerator(randgen_);
    }

    void normal(RealType* outputPtr, std::size_t n, RealType mean, RealType stddev) {
      normal_(outputPtr, n, mean, stddev);
      SafeCudaCall( cudaDeviceSynchronize() );
    }

  private:
    void normal_(float* outputPtr, std::size_t n, float mean, float stddev) {
      curandGenerateNormal(randgen_, outputPtr, n, mean, stddev);
    }

    void normal_(double* outputPtr, std::size_t n, double mean, double stddev) {
      curandGenerateNormalDouble(randgen_, outputPtr, n, mean, stddev);
    }
  };
};

#endif
