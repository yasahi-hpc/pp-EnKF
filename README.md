# Performance portable Ensemble Kalman Filter (pp-LETKF)

[![CI](https://github.com/yasahi-hpc/executor_testing/actions/workflows/test_on_wisteria.yml/badge.svg)](https://github.com/yasahi-hpc/executor_testing/actions)

<div style style=”line-height: 25%” align="center">
<h3> Data assimilation with LETKF on 2D turbulence </h3>
<img src=docs/figs/DA-anime-short-100.gif>
</div>


pp-LETKF is the performance portable implementation of local ensemble transform Kalman filter (LETKF). 
We use C++ parallel algorithms (stdpar), C++ [senders/receivers](https://github.com/NVIDIA/stdexec) and [_mdspan_](https://github.com/kokkos/mdspan) for performance, portablity and productivitiy (P3).
Highly optimized CUDA version is found at [LBM2D-LETKF](https://github.com/hasegawa-yuta-jaea/LBM2D-LETKF).
For questions or comments, please find us in the [AUTHORS](AUTHORS) file.

# Usage
## Preparation
Firstly, you need to git clone on your environment as
```
git clone --recursive git@github.com:yasahi-hpc/executor_testing.git
```
This software relies on external libraries including [eigen](https://gitlab.com/libeigen/eigen), [json](https://github.com/nlohmann/json) and [googletest](https://github.com/google/googletest) (optional, for unit testing). These libraries are included as submodules. CUDA-Aware-MPI or ROCm-Aware-MPI are also needed for NVIDIA and AMD GPUs. In the following, we assume that MPI libraries are appropriately installed.

## Settings
In order to try this repo, following setups are needed.
1. Clone this repo
```
git clone --recursive https://github.com/yasahi-hpc/executor_testing.git
```
2. Download and install the [NVIDIA HPC SDK starting with 22.11](https://developer.nvidia.com/nvidia-hpc-sdk-releases)

## Requirements
This software relies on external libraries including [`stdexec`](https://github.com/NVIDIA/stdexec), [eigen](https://gitlab.com/libeigen/eigen), [json](https://github.com/nlohmann/json), `mdspan` (included in SDK22.11) and [googletest](https://github.com/google/googletest) (optional, for unit testing). These libraries are included as submodules. CUDA-Aware-MPI or ROCm-Aware-MPI are also needed for NVIDIA and AMD GPUs. In the following, we assume that MPI libraries are appropriately installed.

For compilers and CMake, we need the following:
* gcc11+ (c++20 support is necessary)
* nvc++22.11+ (for NVIDIA GPUs)
* CMake3.22.1+

## Configure and Build
```bash
cd executor_testing
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ -DBACKEND=[CUDA, OPENMP] ..
cmake --build . -j 8
```

## Run
We perform an observing system simulation experiment (OSSE) for 2D forced turbulence simulation with Lattice Boltzmann Methods (LBM).
To try data assimilation (DA), you may compare the twin simulations with and without DA.
For DA, you can use nudging or LETKF. LETKF needs at least 4 ensembles (4 GPUs) to work appropriately.
As found in the [sample job scripts](wk), you will need to run nature simulation followd by DA simulations. 
We can perform the OSSE of LBM2D-LETKF in the following manner:
1. Create observation data with Nature run
```bash
cd wk
mpiexec -machinefile $PJM_O_NODEINF -np 1 -npernode 1 \
  ../build/mini-apps/lbm2d-letkf/lbm2d-letkf-stdpar --filename <your_input_file>.json
```

The results would be stored in C++ binary files as follows. The meanings of `base_dir` and `case_name` are found in Input json file section.
```
<base_dir/case_name>/
 |
 |--calc
 |  └──ens0000
 |       |--<name>_step0000000000.dat
 |       |--<name>_step0000000200.dat
 |       |--...
 |
 └──observed/
    └──ens0000
         |--<name>_step0000000000.dat
         |--<name>_step0000000200.dat
         |--...
```
For physical quantities, we store rho (density), u (velocity along x direction), v (velocity along y direction) and vor (vorcitity). The results under `calc` are only used as reference which are not accessed for DA.

2. DA simulation
You can perform LETKF with four ensembles by
```bash
cd wk
mpiexec -machinefile $PJM_O_NODEINF -np 4 -npernode 4 \
  ../build/mini-apps/lbm2d-letkf/lbm2d-letkf-stdpar --filename <your_input_file>.json
```

The results would be stored in C++ binary as follows.
```
<base_dir/case_name>/
 |
 |--calc
 |  |--ens0000
 |  |--ens0001
 |  |--...
 |
 |
 └──observed/
    |--ens0000
    |--ens0001
    |--...
```

## Input json file
Input parameters are given in json format like
```json
{
    "Physics": {
        "rho_ref": 1.0,
        "u_ref": 1.0,
        "nu": 1.0e-4,
        "friction_rate": 5.0e-4,
        "kf": 4.0,
        "fkf": 5.6,
        "dk": 10,
        "sigma": 5,
        "p_amp": 0.01,
        "obs_error_rho": 0.01,
        "obs_error_u": 0.1
    },
    "Settings": {
        "base_dir": "<path-to-the-result-directory>",
        "sim_type": "letkf",
        "case_name": "letkf256",
        "in_case_name": "nature256",
        "nx": 256,
        "ny": 256,
        "spinup": 10000,
        "nbiter": 10000,
        "io_interval": 20,
        "da_interval": 20,
        "obs_interval": 1,
        "lyapnov": false,
        "les": true,
        "da_nud_rate": 0.1,
        "beta": 1.0,
        "rloc_len": 1
    }
}
```
Basically, the json file consists of two major categories ```Physics``` and ```Settings```.  
The former gives the physical settings for simulation.
The latter corresponds to the numerical settings for simulation.

## ```Physics```
| Variable Name | Type | Explanations | 
| --- | --- | --- |
| `rho_ref` | Float | Refernce density. We do not often change this. |
| `u_ref` | Float | Refernce velocity. We do not often change this. |
| `nu` | Float | Kinetic viscosity. |
| `friction_rate` | Float | Friction force strength (acting on large scale vortices). |
| `kf` | Float | Narrow-band injection wavenumbers (wavenumber). |
| `fkf` | Float | Narrow-band injection wavenumbers (strength). |
| `dk` | Int | Narrow-band injection wavenumbers (bandwidth). |
| `sigma` | Int | Narrow-band injection wavenumbers (normalization). |
| `p_amp` | Float | Force term strength. |
| `obs_error_rho` | Float | Observation error strength of density. |
| `obs_error_u` | Float | Observation error strength of velocity (u and v). |

## ```Settings```
| Variable Name | Type | Explanations | 
| --- | --- | --- |
| `base_dir` | String | The simulation results will be stored under this directory. |
| `sim_type` | String | One of `nature`, `nudging`, `letkf` and `no_da`. |
| `case_name` | String | The results are stored in `<base_dir/case_name>` (automatically created if not exists). |
| `in_case_name` | String | The directory including the reference nature run results `<base_dir/in_case_name>`. |
| `nx`, `ny` | Int | The number of grids in x and y directions. |
| `spinup` | Int | The number of time steps for spinup. |
| `nbiter` | Int | The number of time steps for simulation. |
| `io_interval` | Int | The interval of io in time steps. I/O performed for every `io_interval` time steps. |
| `da_interval` | Int | The interval of da in time steps. DA performed for every `da_interval` time steps. |
| `obs_interval` | Int | The spatial intervals of observation. Observation performed for every `obs_interval` grid points. |
| `lyapnov` | Bool | Whether to evaluate the lyapnov exponent of the turbulence. |
| `les` | Bool | Whether to perform LES or DNS. |
| `da_nud_rate` | Float | Nudging coefficient (`0 <= da_nud_rate <= 1`). Used only for Nudging simulation. |
| `beta` | Float | Inflation coefficient. Used only for LETKF simulation. |
| `rloc_len` | Float | Number of adjacent grid points used in R-Localization. Used only for LETKF simulation. |

# Citation
## LETKF
```bibtex
@inproceedings{Hasegawa2022-ScalAH22,
   author={Hasegawa, Yuta and Imamura, Toshiyuki and Ina, Takuya and Onodera, Naoyuki and Asahi, Yuuichi and Idomura, Yasuhiro},
   booktitle={2022 IEEE/ACM Workshop on Latest Advances in Scalable Algorithms for Large-Scale Heterogeneous Systems (ScalAH)}, 
   title={GPU Optimization of Lattice Boltzmann Method with Local Ensemble Transform Kalman Filter}, 
   year={2022},
   volume={},
   number={},
   pages={10-17},
   doi={10.1109/ScalAH56622.2022.00007}
 }
 
 @article{Hasegawa202x,
    author={Hasegawa, Yuta and Onodera, Naoyuki and Asahi, Yuuichi and Ina, Takuya and Idomura, Yasuhiro and Imamura, Toshiyuki},
    journal={in preparation},
    title={Data assimilation of two-dimensional turbulence based on ensemble Kalman filter with spatially sparse and noisy observation}
 }
```
