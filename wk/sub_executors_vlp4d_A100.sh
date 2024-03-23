#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

. /etc/profile.d/modules.sh # Initialize module command

module purge
module load nvidia/24.1 cmake/3.24.0
export NVLOCALRC=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/compilers/bin/localrc_gcc12.2.0

# Need GPUs to build the code appropriately
# So compile inside a batch job, wherein GPUs are visible
if [ ! -d "../build" ]
then
    cd ../
    rm -rf build
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=nvc++ \
          -DBACKEND=CUDA \
          -DAPPLICATION=vlp4d \
          ..
    cmake --build . -j 8
    cd ../wk/
fi

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no
export OMPI_MCA_plm_rsh_agent=/bin/pjrsh
export OMP_NUM_THREADS=36

../build/mini-apps/vlp4d/executors/vlp4d-executors SLD10_large.dat