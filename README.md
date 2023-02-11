# Testing Executors

## Examples
We added small examples under tutorial and mini-apps. 
* [tutorial](tutorial)
* [mini-apps](mini-apps)

## Settings
In order to try this repo, following setups are needed.
1. Clone this repo
```
git clone --recursive https://github.com/yasahi-hpc/executor_testing.git
```
2. Download and install the [NVIDIA HPC SDK starting with 22.11](https://developer.nvidia.com/nvidia-hpc-sdk-releases)

## Requirements
The examples in this repo depend on [`stdexec`](https://github.com/NVIDIA/stdexec) and `mdspan` (included in SDK22.11). 
More specifically, we need the newer compilers and cmake.
* gcc11+ (c++20 support is necessary)
* nvc++22.11+ (for NVIDIA GPUs)
* CMake3.22.1+

## Tested environment
### SGI8600 (NVIDIA V100)
On SGI8600, we need to load following modules

1. Environment
```bash
module purge
spack load gcc@11.3.0 cmake
module load nvhpc/22.11 hpcx-ompi
```

2. Build
```bash
cd executor_testing
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ ..
cmake --build . -j 8
```

### Wisteria (A100)
On Wisteria, we need to load following modules

1. Environment
```bash
module purge
spack_env
spack load gcc@11.3.0
spack load cmake@3.24.3
module load nvhpc/22.11 hpcx-ompi
```

2. Build
```bash
cd executor_testing
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ ..
cmake --build . -j 8
```
