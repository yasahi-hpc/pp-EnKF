#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=2

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
    mkdir build && cd build
    cmake -DAdaptiveCpp_DIR=/work/04/jh220031a/i18048/lib/local/AdaptiveCpp/lib/cmake/AdaptiveCpp -DACPP_TARGETS="cuda-nvcxx:sm_80" \
          -DPROGRAMMING_MODEL=SYCL \
          -DBACKEND=CUDA \
          -DAPPLICATION=heat3d-mpi \
          ..
    cmake --build . -j 8
    cd ../wk/
fi

echo "sycl"
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 2 \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/sycl/heat3d-mpi-sycl --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0