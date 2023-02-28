#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
  #include "Cuda_Transpose.hpp"
#else
  #include "OpenMP_Transpose.hpp"
#endif

#endif
