#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

. /etc/profile.d/modules.sh # Initialize module command

module purge

# Load spack
export HOME=/work/jh220031a/i18048
. $HOME/spack/share/spack/setup-env.sh

spack load gcc@11.3.0
spack load cmake@3.24.3
module use /work/jh220031a/i18048/lib/nvidia/hpc_sdk22.11/modulefiles
module use /work/jh220031a/i18048/lib/nvidia/hpc_sdk22.11/Linux_x86_64/22.11/comm_libs/hpcx/latest/modulefiles
module load nvhpc/22.11
#module load nvhpc/22.11 hpcx-ompi

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 1 \
    ../build/mini-apps/heat3d-mpi/stdpar/heat3d-mpi-stdpar --px 1 --py 1 --pz 1 --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 1 \
    ../build/mini-apps/heat3d-mpi/executors/heat3d-mpi-exec --px 1 --py 1 --pz 1 --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
