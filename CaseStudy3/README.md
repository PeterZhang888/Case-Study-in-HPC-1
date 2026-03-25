README
======

This folder contains the MATLAB code, figures, and saved results for the Conjugate Gradient (CG) assignment.

Contents
--------

1. MATLAB function files

- `Q1CG.m`  
  MATLAB function implementing the Conjugate Gradient (CG) method for solving
  \[
  Ax=b
  \]
  where \(A\) is symmetric positive definite.  
  Outputs:
  - approximate solution `x`
  - residual norm history `res_norm`
  - number of iterations `iter`

- `get_covariance_matrix.m`  
  MATLAB function for generating a symmetric positive definite Matérn covariance matrix \(A\) and a random right-hand side vector \(b\).  
  Inputs:
  - matrix size `N`
  - kernel parameter `tau`
  - noise level `noise`

2. MATLAB scripts for each question

- `Q2.m`  
  Script for Question 2.  
  It runs the CG method for three matrix sizes:
  - \(N = 512\)
  - \(N = 1024\)
  - \(N = 2048\)

  using:
  - `tau = 20`
  - `noise = 0.005`

  The script records:
  - the solution vector
  - the number of iterations
  - the relative residual norm history
  - the final relative residual norm

  It also produces the convergence plot:
  - `CG_convergence_for_Matern_covariance_matrices.png`

  and saves the numerical results in:
  - `Q2_results.mat`

- `Q3.m`  
  Script for Question 3.  
  It uses:
  - \(N = 1024\)
  - `tau = 416` (last three digits of the student ID)

  and varies the noise level from `0.5` down to `5e-9`.

  The script records:
  - the number of CG iterations
  - the condition number of \(A\)

  It produces the plots:
  - `CG_iterations_versus_noise_level.png`
  - `Condition_number_versus_noise_level.png`

- `Q4.m`  
  Script for Question 4.  
  It uses:
  - `N = 1024`
  - `tau = 100`
  - `noise = 0.05`

  It compares the CG convergence for:
  - the original matrix \(A\)
  - the transformed matrix \(A_{\text{new}}\)

  The script records:
  - iteration counts
  - condition numbers
  - final relative residual norms

  It produces the plots:
  - `Residual_Comparison.png`
  - `Eigenvalue_Comparison.png`

  and saves numerical outputs in:
  - `Q4_results.mat`

3. Saved result files

- `Q2_results.mat`  
  Saved MATLAB data file containing the numerical results for Question 2.

- `Q4_results.mat`  
  Saved MATLAB data file containing the numerical results for Question 4.

4. Figure files

- `CG_convergence_for_Matern_covariance_matrices.png`  
  Convergence curves for the CG method for different matrix sizes in Question 2.

- `CG_iterations_versus_noise_level.png`  
  Plot of CG iteration count against noise level for Question 3.

- `Condition_number_versus_noise_level.png`  
  Plot of matrix condition number against noise level for Question 3.

- `Residual_Comparison.png`  
  Comparison of CG residual convergence for \(A\) and \(A_{\text{new}}\) in Question 4.

- `Eigenvalue_Comparison.png`  
  Comparison of the eigenvalue spectra of \(A\) and \(A_{\text{new}}\) in Question 4.

How to run
----------

Place all files in the same MATLAB working directory and run the scripts individually:

```matlab
Q1CG
Q2
Q3
Q4
