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
spack load cmake@3.24.3%gcc@8.3.1

#module load /work/04/jh220031a/i18048/lib/nvidia/hpc_sdk22.11/modulefiles/nvhpc/22.11
module load /work/04/jh220031a/i18048/lib/nvidia/hpc_sdk23.3/modulefiles/nvhpc/23.3
#module load /work/jh220031a/i18048/lib/nvidia/hpc_sdk22.11/modulefiles/nvhpc/22.11

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

../build/tutorial/07_heat2d_repeat_n/07_heat2d_repeat_n
../build/tutorial/06_sender_from_function/06_sender_from_function
../build/tutorial/05_heat2d/05_heat_test
../build/mini-apps/heat3d/stdpar/heat3d-stdpar
#../build/tutorial/04_stream/04_stream_test
#../build/tutorial/02_hello_world_nvexec/02_hello_world
