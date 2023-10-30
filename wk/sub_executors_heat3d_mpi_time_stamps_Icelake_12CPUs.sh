#!/bin/bash
#PJM -L "node=6"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=60:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=12

. /etc/profile.d/modules.sh # Initialize module command

module purge

# Load spack
export HOME=/work/jh220031a/i18048
. $HOME/spack/share/spack/setup-env.sh

module load nvidia/23.3 cmake/3.24.0 nvmpi/23.3
export NVLOCALRC=/work/opt/local/x86_64/cores/nvidia/23.3/Linux_x86_64/23.3/compilers/bin/localrc_gcc12.2.0

# Need GPUs to build the code appropriately
# So compile inside a batch job, wherein GPUs are visible
if [ ! -d "../build" ]
then
    cd ../
    rm -rf build
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=nvc++ -DBACKEND=OPENMP -DCMAKE_CXX_FLAGS="-std=c++20" ..
    cmake --build . -j 8
    cd ../wk/
fi

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no
export OMPI_MCA_plm_rsh_agent=/bin/pjrsh
export OMP_NUM_THREADS=36

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC --bind-to none \
    ../build/mini-apps/heat3d-mpi/executors/heat3d-mpi-executors --px 1 --py 1 --pz 12 --nx 1536 --ny 1536 --nz 128 --nbiter 100 --freq_diag 0 --use_time_stamps 1
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC --bind-to none \
    ../build/mini-apps/heat3d-mpi/executors/heat3d-mpi-executors --px 1 --py 1 --pz 12 --nx 1536 --ny 1536 --nz 128 --nbiter 100 --freq_diag 0 --use_time_stamps 1 --is_async 1
