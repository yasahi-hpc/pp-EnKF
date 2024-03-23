#!/bin/bash -e
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=30:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=4

. /etc/profile.d/modules.sh # Initialize module command

module purge
module load nvidia/24.1 cmake/3.24.0
export NVLOCALRC=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/compilers/bin/localrc_gcc12.2.0

# Need GPUs to build the code appropriately
# So compile inside a batch job, wherein GPUs are visible
if [ ! -d "../build_CUDA" ]
then
    cd ../
    rm -rf build_CUDA
    mkdir build_CUDA && cd build_CUDA
    cmake -DCMAKE_CXX_COMPILER=nvc++ -DBACKEND=CUDA -DBUILD_TESTING=ON ..
    cmake --build . -j 8
    cd ../wk/
fi

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_RNDV_FRAG_MEM_TYPE=cuda

../build_CUDA/tests/executors/google-tests-executors
../build_CUDA/tests/stdpar/google-tests-stdpar

touch success_CUDA
