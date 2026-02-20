#!/bin/bash
#SBATCH -J TSQR_4nodes
#SBATCH -o ./casestudy1.out
#SBATCH -e ./casestudy1.err
#SBATCH --partition=compute
#SBATCH --ntasks=4

cd $SLURM_SUBMIT_DIR

module load tbb/2021.12 compiler-rt/2024.1.0 mkl/2024.1 gcc/13.2.0-gcc-8.5.0-sfeapnb mpi

mpirun -np 4 ./casestudy1Exec
