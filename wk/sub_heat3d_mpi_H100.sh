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

echo "stdpar"
mpirun -npernode 2 -n 2 -x LD_LIBRARY_PATH \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/stdpar/heat3d-mpi-stdpar --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0

echo "thrust"
mpirun -npernode 2 -n 2 -x LD_LIBRARY_PATH \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/thrust/heat3d-mpi-thrust --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0

echo "executors"
mpirun -npernode 2 -n 2 -x LD_LIBRARY_PATH \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/executors/heat3d-mpi-executors --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0

echo "executors (async)"
mpirun -npernode 2 -n 2 -x LD_LIBRARY_PATH \
    ./wrapper.sh ../build/mini-apps/heat3d-mpi/executors/heat3d-mpi-executors --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0 --is_async 1
