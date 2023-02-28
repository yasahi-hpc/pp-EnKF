#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
  #include "Cuda_FFT.hpp"
#else
  #include "OpenMP_FFT.hpp"
#endif

#endif
