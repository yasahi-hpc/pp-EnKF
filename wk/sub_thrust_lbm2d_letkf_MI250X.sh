#!/bin/bash
#SBATCH -A CFD173
#SBATCH -J lbm2d
#SBATCH -o %x-%j.out
#SBATCH -t 00:60:00
#SBATCH -p batch
#SBATCH -N 1

module purge
module load PrgEnv-amd
module load amd/5.4.3
module load cray-mpich/8.1.26
module load craype-accel-amd-gfx90a
module load cmake/3.23.2
module list

# Build
#cd ..
#if [ -d "build" ]
#then
#    rm -rf build
#fi
#
#mkdir build && cd build
#cmake -DCMAKE_CXX_COMPILER=hipcc -DBACKEND=HIP -DCMAKE_EXE_LINKER_FLAGS="${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" ..
#cmake --build . -j 8
#cd ../wk

# nature run (1 MPI)
#export MPICH_GPU_SUPPORT_ENABLED=1
#OMP_NUM_THREADS=7 srun -N1 -n1 -c7 --ntasks-per-node=1 ../build/mini-apps/lbm2d-letkf/thrust/lbm2d-letkf-thrust --filename nature_512.json

# lbm2d-letkf (4 MPI)
export MPICH_GPU_SUPPORT_ENABLED=1
OMP_NUM_THREADS=7 srun -N1 -n4 -c7 --ntasks-per-node=4 ../build/mini-apps/lbm2d-letkf/thrust/lbm2d-letkf-thrust --filename letkf_512.json
