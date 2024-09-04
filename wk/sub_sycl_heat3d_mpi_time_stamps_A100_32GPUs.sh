#!/bin/bash
#PJM -L "node=4"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220036a
#PJM --mpi proc=32

. /etc/profile.d/modules.sh # Initialize module command

module purge
module load nvidia/24.1 cmake/3.24.0

export NVLOCALRC=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/compilers/bin/localrc_gcc12.2.0
export AdaptiveCpp_ROOT=/work/04/jh230064a/i18048/lib/AdaptiveCpp
export Boost_ROOT=/work/04/jh230064a/i18048/lib/boost_1_85_0
export PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/hpcx/bin/:$PATH
export LD_LIBRARY_PATH=${Boost_ROOT}/lib/:$LD_LIBRARY_PATH
export ACPP_NVCXX=`which nvc++`
export ACPP_RT_MAX_CACHED_NODES=0

#export UCX_MEMTYPE_CACHE=n
#export UCX_RNDV_FRAG_MEM_TYPE=cuda
export OMPI_MCA_plm_rsh_agent=/bin/pjrsh

# Need GPUs to build the code appropriately
# So compile inside a batch job, wherein GPUs are visible
if [ ! -d "../build" ]
then
    cd ../
    mkdir build && cd build
    cmake -DACPP_TARGETS="cuda-nvcxx:sm_80" \
          -DCMAKE_PREFIX_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/hpcx/ \
          -DPROGRAMMING_MODEL=SYCL \
          -DBACKEND=CUDA \
          -DAPPLICATION=heat3d-mpi \
          ..
    cmake --build . -j 8
    cd ../wk/
fi

echo "sync"
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/sycl/heat3d-mpi-sycl --px 1 --py 1 --pz 32 --nx 1536 --ny 1536 --nz 48 --nbiter 1000 --freq_diag 0 --use_time_stamps 1

echo "async"
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/sycl/heat3d-mpi-sycl --px 1 --py 1 --pz 32 --nx 1536 --ny 1536 --nz 48 --nbiter 1000 --freq_diag 0 --use_time_stamps 1 --is_async 1