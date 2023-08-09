#!/bin/bash
#SBATCH -A CFD173
#SBATCH -J heat3d
#SBATCH -o %x-%j.out
#SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -N 1

module purge
module load PrgEnv-amd
module load amd/5.4.3
module load cray-mpich/8.1.26
module load craype-accel-amd-gfx90a
module load cmake/3.23.2
module list

../build/tests/executors/google-tests-executors
