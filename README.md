# Testing Executors

## Settings
First of all, you need to clone this repository.
```
git clone --recursive https://github.com/yasahi-hpc/executor_testing.git
```

### SGI8600
On SGI8600, we need to load following modules

1. Environment
```bash
module purge
spack load gcc@11.3.0 cmake
module load nvhpc/22.11
```

2. Build
```bash
cd executor_testing
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ ..
cmake --build . -j 8
```

### Wisteria
On Wisteria, we need to load following modules

1. Environment
```bash
module purge
spack load gcc@11.2.0 cmake@3.23.1%gcc@11.2.0
module load nvhpc/22.11
```

2. Build
```bash
cd executor_testing
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ ..
cmake --build . -j 8
```
