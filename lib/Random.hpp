#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#if defined(ENABLE_OPENMP)
  #include "Cuda_Random.hpp"
#elif defined(_NVHPC_CUDA) || defined(__CUDACC__)
  #include "Cuda_Random.hpp"
#else
  #include "Cuda_Random.hpp"
#endif

#endif
