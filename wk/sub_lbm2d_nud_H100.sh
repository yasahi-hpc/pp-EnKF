#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -N serial

. /etc/profile.d/modules.sh # Initialize module command

module purge
module load nvhpc/24.1 cmake/3.28.3
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

export UCX_RNDV_FRAG_MEM_TYPE=cuda

echo "nature"
mpirun -npernode 1 -n 1 -x LD_LIBRARY_PATH \
    ../build/mini-apps/lbm2d-letkf/stdpar/lbm2d-letkf-stdpar --filename nature_256.json

echo "nudging (stdpar)"
mpirun -npernode 1 -n 1 -x LD_LIBRARY_PATH \
    ../build/mini-apps/lbm2d-letkf/stdpar/lbm2d-letkf-stdpar --filename nudging_256.json

echo "nudging (thrust)"
mpirun -npernode 1 -n 1 -x LD_LIBRARY_PATH \
    ../build/mini-apps/lbm2d-letkf/thrust/lbm2d-letkf-thrust --filename nudging_256.json

echo "nudging (executors)"
mpirun -npernode 1 -n 1 -x LD_LIBRARY_PATH \
    ../build/mini-apps/lbm2d-letkf/executors/lbm2d-letkf-executors --filename nudging_256.json
