#ifndef __MPI_UTILS_HPP__
#define __MPI_UTILS_HPP__

#include <mpi.h>

#if defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__)
    #include <thrust/complex.h>
    template <typename RealType> using Complex = thrust::complex<RealType>;
#else
    #include <complex>
    template <typename RealType> using Complex = std::complex<RealType>;
#endif

namespace Impl {
  template <typename T,
          std::enable_if_t<std::is_same_v<T, int             > ||
                           std::is_same_v<T, float           > ||
                           std::is_same_v<T, double          > ||
                           std::is_same_v<T, Complex<float>  > ||
                           std::is_same_v<T, Complex<double> >
                           , std::nullptr_t> = nullptr
  >
  MPI_Datatype getMPIDataType() {
    MPI_Datatype type;
    if(std::is_same_v<T, int             >) type = MPI_INT;
    if(std::is_same_v<T, float           >) type = MPI_FLOAT;
    if(std::is_same_v<T, double          >) type = MPI_DOUBLE;
    if(std::is_same_v<T, Complex<float>  >) type = MPI_COMPLEX;
    if(std::is_same_v<T, Complex<double> >) type = MPI_DOUBLE_COMPLEX;
  
    return type;
  }
};

#endif
