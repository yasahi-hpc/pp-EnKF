#ifndef __STDPAR_LINALG_HPP__
#define __STDPAR_LINALG_HPP__

#if defined(ENABLE_OPENMP) 
  #include "../cuda_linalg.hpp"
#elif defined(_NVHPC_CUDA) || defined(__CUDACC__) 
  #include "../cuda_linalg.hpp"
#else
  #include "../cuda_linalg.hpp"
#endif

#endif
