#!/bin/bash
set -euo pipefail

module purge
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load mkl/2024.1
module load gcc/13.2.0-gcc-8.5.0-sfeapnb
module load mpi

mpicc -O3 -o casestudy1bExec casestudy1b.c \
  -I"${MKLROOT}/include" -L"${MKLROOT}/lib" \
  -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl
