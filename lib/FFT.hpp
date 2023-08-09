#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined(ENABLE_OPENMP) 
  #include "OpenMP_FFT.hpp"
#elif defined(_NVHPC_CUDA) || defined(__CUDACC__) 
  #include "Cuda_FFT.hpp"
#elif defined(__HIPCC__)
  #include "HIP_FFT.hpp"
#else
  #include "OpenMP_FFT.hpp"
#endif

#endif
