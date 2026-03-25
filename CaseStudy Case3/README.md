README
======

This folder contains the MATLAB code, figures, saved results, and report materials for the Conjugate Gradient (CG) assignment.

Submission structure
--------------------

The submission is organised as follows:

- `main.m`  
  Main MATLAB script containing the overall demonstrations and solutions for Questions 2, 3, and 4.  
  This is the main file that should be run first.

- `Q1CG.m`  
  MATLAB function implementing the Conjugate Gradient method.

- `get_covariance_matrix.m`  
  MATLAB function for generating the symmetric positive definite Matérn covariance matrix and right-hand side vector.

- `Q2.m`  
  Script for Question 2.

- `Q3.m`  
  Script for Question 3.

- `Q4.m`  
  Script for Question 4.

- `Q2_results.mat`  
  Saved numerical results for Question 2.

- `Q4_results.mat`  
  Saved numerical results for Question 4.

- `CG_convergence_for_Matern_covariance_matrices.png`  
  Figure produced in Question 2.

- `CG_iterations_versus_noise_level.png`  
  Figure produced in Question 3.

- `Condition_number_versus_noise_level.png`  
  Figure produced in Question 3.

- `Residual_Comparison.png`  
  Figure produced in Question 4.

- `Eigenvalue_Comparison.png`  
  Figure produced in Question 4.

- `CG_report_Q2_Q4.tex`  
  LaTeX report discussing the numerical results for Questions 2, 3, and 4.

File descriptions
-----------------

1. Main script

`main.m` runs the scripts for Questions 2, 3, and 4 in sequence, so that all demonstrations and numerical results can be reproduced from a single MATLAB file.

2. Function files

`Q1CG.m` solves the linear system
\[
Ax = b
\]
using the Conjugate Gradient method, where \(A\) is symmetric positive definite. It returns:
- the approximate solution vector
- the residual norm history
- the number of iterations

`get_covariance_matrix.m` generates a Matérn covariance matrix \(A\) together with a random right-hand side vector \(b\), using the specified values of:
- matrix dimension \(N\)
- kernel parameter \(\tau\)
- noise level

3. Question scripts

`Q2.m` investigates how the dimension of the covariance matrix affects CG convergence. It uses:
- \(\tau = 20\)
- noise \(= 0.005\)
- matrix sizes \(N = 512, 1024, 2048\)

It records:
- the number of iterations
- the relative residual norm history
- the final relative residual norm

`Q3.m` investigates how the noise level affects the condition number of \(A\) and the convergence of CG. It uses:
- \(N = 1024\)
- \(\tau = 416\) (from the last three digits of the student ID)

It varies the noise level from `0.5` down to `5e-9`, and records:
- the number of CG iterations
- the condition number of \(A\)

`Q4.m` compares the convergence of CG for the original matrix \(A\) and the transformed matrix \(A_{\text{new}}\). It uses:
- \(N = 1024\)
- \(\tau = 100\)
- noise \(= 0.05\)

It records:
- iteration counts
- condition numbers
- final relative residual norms

It also compares:
- residual convergence
- eigenvalue spectra

How to run the code
-------------------

1. Place all files in the same MATLAB working directory.

2. Open MATLAB and set the current folder to this directory.

3. Run:

```matlab
main
