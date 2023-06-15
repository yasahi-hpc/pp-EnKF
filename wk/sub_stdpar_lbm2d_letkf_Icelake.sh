#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=60:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=4
#PJM --omp thread=18

. /etc/profile.d/modules.sh # Initialize module command

module purge

# Load spack
export HOME=/work/jh220031a/i18048
. $HOME/spack/share/spack/setup-env.sh

spack load gcc@11.3.0
spack load cmake@3.24.3%gcc@8.3.1
module load /work/04/jh220031a/i18048/lib/nvidia/hpc_sdk23.3/modulefiles/nvhpc/23.3
module list

# Need GPUs to build the code appropriately
# So compile inside a batch job, wherein GPUs are visible
if [ ! -d "../build" ]
then
    cd ../
    rm -rf build
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=nvc++ -DBACKEND=OPENMP ..
    cmake --build . -j 8
    cd ../wk/
fi

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

mpiexec -machinefile $PJM_O_NODEINF -np 1 -npernode 1 \
    ../build/mini-apps/lbm2d-letkf/stdpar/lbm2d-letkf-stdpar --filename nature.json

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode $PJM_MPI_PROC \
    ../build/mini-apps/lbm2d-letkf/stdpar/lbm2d-letkf-stdpar --filename letkf.json
