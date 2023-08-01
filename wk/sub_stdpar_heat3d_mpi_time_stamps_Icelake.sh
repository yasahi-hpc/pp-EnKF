#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=60:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=2

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
export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC --map-by ppr:1:socket:PE=36 \
#    ../build/mini-apps/heat3d-mpi/stdpar/heat3d-mpi-stdpar --px 1 --py 1 --pz 1 --nx 512 --ny 512 --nz 512 --nbiter 100 --freq_diag 0 --use_time_stamps 1
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC --map-by ppr:1:socket:PE=36 \
    ../build/mini-apps/heat3d-mpi/stdpar/heat3d-mpi-stdpar --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 100 --freq_diag 0 --use_time_stamps 1
