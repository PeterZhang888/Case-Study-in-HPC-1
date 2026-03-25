README
======

This folder contains the MATLAB code, figures, saved results, and report materials for the Conjugate Gradient (CG) assignment.

Submission structure
--------------------

The submission is organised as follows:

- `main.m`  
  Main MATLAB script containing all demonstrations and solutions for Questions 2, 3, and 4.  
  This is the primary file to run.

- `Q1CG.m`  
  MATLAB function implementing the Conjugate Gradient method.

- `get_covariance_matrix.m`  
  MATLAB function for generating the symmetric positive definite Matérn covariance matrix and right-hand side vector.

- `Q2_results.mat`  
  Saved numerical results for Question 2.

- `Q4_results.mat`  
  Saved numerical results for Question 4.

- `CG_convergence_for_Matern_covariance_matrices.png`  
  Figure produced for Question 2.

- `CG_iterations_versus_noise_level.png`  
  Figure produced for Question 3.

- `Condition_number_versus_noise_level.png`  
  Figure produced for Question 3.

- `Residual_Comparison.png`  
  Figure produced for Question 4.

- `Eigenvalue_Comparison.png`  
  Figure produced for Question 4.

- `CG_report_Q2_Q4.tex`  
  LaTeX report discussing the numerical results for Questions 2, 3, and 4.

File descriptions
-----------------

1. Main script

`main.m` contains all computations and demonstrations required for Questions 2, 3, and 4.  
It performs:
- generation of covariance matrices
- execution of the Conjugate Gradient method
- computation of residuals and condition numbers
- generation of plots and saved outputs

2. Function files

`Q1CG.m` solves the linear system
\[
Ax = b
\]
using the Conjugate Gradient method for symmetric positive definite matrices. It returns:
- the approximate solution
- the residual norm history
- the number of iterations

`get_covariance_matrix.m` generates a Matérn covariance matrix \(A\) and a random vector \(b\), based on:
- matrix size \(N\)
- kernel parameter \(\tau\)
- noise level

3. Numerical experiments

Question 2:
- investigates the effect of matrix size on CG convergence  
- uses \(N = 512, 1024, 2048\), \(\tau = 20\), noise \(= 0.005\)

Question 3:
- investigates the effect of noise on conditioning and convergence  
- uses \(N = 1024\), \(\tau = 416\)  
- varies noise from \(0.5\) to \(5 \times 10^{-9}\)

Question 4:
- compares CG convergence for the original matrix \(A\) and the transformed matrix \(A_{\text{new}}\)  
- analyses residual behaviour and eigenvalue distributions

How to run the code
-------------------

1. Place all files in the same MATLAB working directory.

2. Open MATLAB and set the current folder to this directory.

3. Run:

```matlab
main
