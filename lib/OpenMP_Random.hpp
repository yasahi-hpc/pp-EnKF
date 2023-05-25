#ifndef __OPENMP_RANDOM_HPP__
#define __OPENMP_RANDOM_HPP__

#include <random>
#include <type_traits>

namespace Impl {
  template <typename RealType,
            typename std::enable_if<std::is_same<RealType, float>::value ||
                                    std::is_same<RealType, double>::value 
                                   >::type * = nullptr>
  struct Random {
  private:
    std::default_random_engine generator_;

  public:
    Random() {}

    ~Random() {}

    void normal(RealType* outputPtr, std::size_t n, RealType mean, RealType stddev) {
      normal_(outputPtr, n, mean, stddev);
    }

  private:
    void normal_(float* outputPtr, std::size_t n, float mean, float stddev) {
      std::normal_distribution<float> distribution(mean, stddev);
      for(std::size_t i=0; i<n; i++) {
        outputPtr[i] = distribution(generator_);
      }
    }

    void normal_(double* outputPtr, std::size_t n, double mean, double stddev) {
      std::normal_distribution<double> distribution(mean, stddev);
      for(std::size_t i=0; i<n; i++) {
        outputPtr[i] = distribution(generator_);
      }
    }
  };
};

#endif
