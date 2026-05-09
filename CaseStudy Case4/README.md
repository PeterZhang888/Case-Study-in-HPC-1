# Case Study 4: Multigrid Assignment

This repository contains my solution for Case Study 4: Multigrid Method.

The code solves the two-dimensional Poisson problem

$$
-\Delta u = f
$$

on the unit square $(0,1)^2$, with homogeneous Dirichlet boundary conditions

$$
u(x,y)=0 \quad \text{for } x=0,\ x=1,\ y=0,\ \text{or } y=1.
$$

The problem is discretised using second-order finite differences on a uniform $n \times n$ interior grid, where

$$
n = 2^p - 1.
$$

## Files

### Poisson2D.py

This file contains the function `Poisson2D(n)`.

It constructs the sparse matrix $A$ for the five-point finite difference discretisation of the two-dimensional Poisson problem. Since there are $n$ interior grid points in each spatial direction, the matrix has size $n^2 \times n^2$.

### Cycle.py

This file contains the functions used for the multigrid V-cycle:

- `weighted_jacobi`
- `restriction_operator`
- `build_multigrid_hierarchy`
- `v_cycle`

The V-cycle uses weighted Jacobi smoothing with relaxation parameter $\omega = 2/3$.

The coarse-grid matrix is built using

$$
A_c = R A_f P.
$$

### Test.ipynb

This Jupyter notebook contains the demonstration and numerical test.

It imports the functions from `Poisson2D.py` and `Cycle.py`, generates a random right-hand side vector, starts from the zero initial guess, applies fewer than 15 V-cycles, records the residual norm after each cycle, and plots the convergence result.
