#!/bin/bash -e
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=30:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=4

. /etc/profile.d/modules.sh # Initialize module command

module purge
module load nvidia/23.11 cmake/3.24.0

# Setup Adaptive CPP env
export PATH=/work/opt/local/x86_64/cores/nvidia/23.11/Linux_x86_64/23.11/comm_libs/openmpi4/bin/:$PATH
export ACPP_NVCXX=/work/opt/local/x86_64/cores/nvidia/23.11/Linux_x86_64/23.11/compilers/bin/nvc++

# Need GPUs to build the code appropriately
# So compile inside a batch job, wherein GPUs are visible
if [ ! -d "../build_SYCL" ]
then
    cd ../
    rm -rf build_SYCL
    mkdir build_SYCL && cd build_SYCL
    cmake .. -DPROGRAMMING_MODEL=SYCL \
             -DBACKEND=CUDA \
             -DAdaptiveCpp_DIR=/work/04/jh220031a/i18048/lib/local/AdaptiveCpp/lib/cmake/AdaptiveCpp \
             -DACPP_TARGETS="cuda-nvcxx:sm_80"
    cmake --build . -j 8
    cd ../wk/
fi

../build_SYCL/tests/sycl/google-tests-sycl

touch success_SYCL