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

../build/mini-apps/vlp4d/executors/vlp4d-executors SLD10_large.dat
