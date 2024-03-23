#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=60:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=4

. /etc/profile.d/modules.sh # Initialize module command

module purge
module load nvidia/23.11 cmake/3.24.0

export PATH=/work/opt/local/x86_64/cores/nvidia/23.11/Linux_x86_64/23.11/comm_libs/openmpi4/bin/:$PATH
export ACPP_NVCXX=/work/opt/local/x86_64/cores/nvidia/23.11/Linux_x86_64/23.11/compilers/bin/nvc++

# Need GPUs to build the code appropriately
# So compile inside a batch job, wherein GPUs are visible
if [ ! -d "../build" ]
then
    cd ../
    rm -rf build
    mkdir build && cd build
    cmake -DAdaptiveCpp_DIR=/work/04/jh220031a/i18048/lib/local/AdaptiveCpp/lib/cmake/AdaptiveCpp -DACPP_TARGETS="cuda-nvcxx:sm_80" \
          -DPROGRAMMING_MODEL=SYCL \
          -DBACKEND=CUDA \
          -DAPPLICATION=lbm2d-letkf \
          ..
    cmake --build . -j 8
    cd ../wk/
fi

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_RNDV_FRAG_MEM_TYPE=cuda

mpiexec -machinefile $PJM_O_NODEINF -np 1 -npernode 1 \
    ../build/mini-apps/lbm2d-letkf/sycl/lbm2d-letkf-sycl --filename nature_256.json

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 4 \
    ./wrapper.sh ../build/mini-apps/lbm2d-letkf/sycl/lbm2d-letkf-sycl --filename letkf_256.json

#mpiexec -machinefile $PJM_O_NODEINF -np 1 -npernode 1 \
#    ../build/mini-apps/lbm2d-letkf/sycl/lbm2d-letkf-sycl --filename nature_512.json

#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 4 \
#    ./wrapper.sh ../build/mini-apps/lbm2d-letkf/sycl/lbm2d-letkf-sycl --filename letkf_512.json

#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 4 \
#    ./wrapper.sh ../build/mini-apps/lbm2d-letkf/sycl/lbm2d-letkf-sycl --filename letkf_async_512.json
