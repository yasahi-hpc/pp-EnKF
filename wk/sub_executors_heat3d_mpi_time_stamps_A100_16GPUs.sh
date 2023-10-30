#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=16

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
    cmake -DCMAKE_CXX_COMPILER=nvc++ -DBACKEND=CUDA ..
    cmake --build . -j 8
    cd ../wk/
fi

echo "HOME: ${HOME}"
echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_RNDV_FRAG_MEM_TYPE=cuda
export OMPI_MCA_plm_rsh_agent=/bin/pjrsh

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/executors/heat3d-mpi-executors --px 1 --py 1 --pz 16 --nx 1536 --ny 1536 --nz 96 --nbiter 1000 --freq_diag 0 --use_time_stamps 1
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/executors/heat3d-mpi-executors --px 1 --py 1 --pz 16 --nx 1536 --ny 1536 --nz 96 --nbiter 1000 --freq_diag 0 --use_time_stamps 1 --is_async 1
