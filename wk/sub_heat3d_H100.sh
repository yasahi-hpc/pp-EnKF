#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
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

echo "stdpar"
../build/mini-apps/heat3d/stdpar/heat3d-stdpar --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0

echo "thrust"
../build/mini-apps/heat3d/thrust/heat3d-thrust --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0

echo "executors"
../build/mini-apps/heat3d/executors/heat3d-executors --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
