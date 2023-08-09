#ifndef __LINALG_HPP__
#define __LINALG_HPP__

#if defined(ENABLE_OPENMP) 
  #include "openmp_linalg.hpp"
#elif defined(_NVHPC_CUDA) || defined(__CUDACC__) 
  #include "cuda_linalg.hpp"
#elif defined(__HIPCC__)
  #include "hip_linalg.hpp"
#else
  #include "openmp_linalg.hpp"
#endif

#endif
