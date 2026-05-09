# Case Study 4: Multigrid Assignment

This repository contains my solution for Case Study 4: Multigrid Assignment.

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

## How to Run

Make sure the following three files are in the same folder:

- `Poisson2D.py`
- `Cycle.py`
- `Test.ipynb`

Open the notebook with:

`jupyter notebook Test.ipynb`

or:

`jupyter lab Test.ipynb`

Then run the cells in order.

## Required Packages

The code uses:

- `numpy`
- `scipy`
- `matplotlib`

They can be installed with:

`pip install numpy scipy matplotlib`

## Output

The notebook produces a plot of the residual norm

$$
\|f - Au\|_2
$$

against the number of V-cycles.

A decreasing line on the logarithmic plot shows that the multigrid V-cycle is converging.
