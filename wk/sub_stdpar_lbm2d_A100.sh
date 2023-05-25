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
module load /work/jh220031a/i18048/lib/nvidia/hpc_sdk22.11/modulefiles/nvhpc/22.11

# Need GPUs to build the code appropriately
# So compile inside a batch job, wherein GPUs are visible
cd ../
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ -DBACKEND=CUDA ..
cmake --build . -j 8
cd wk/

../build/mini-apps/lbm2d-letkf/stdpar/lbm2d-letkf-stdpar --filename nature.json
