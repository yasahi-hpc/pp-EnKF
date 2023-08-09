#!/bin/bash
#SBATCH -A CFD173
#SBATCH -J heat3d-mpi
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1

#module purge
module load PrgEnv-amd
module load amd/5.4.3
module load cray-mpich/8.1.26
module load craype-accel-amd-gfx90a
module list

export OMP_NUM_THREADS=7
export MPICH_GPU_SUPPORT_ENABLED=1

srun -N1 -n1 -c7 --ntasks-per-node=1 ../build/mini-apps/heat3d/thrust/heat3d-thrust --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
