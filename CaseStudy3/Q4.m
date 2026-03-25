clear; clc; close all;
N=1024;
tau=100;
noise=0.05;
tol= 1e-8;
max_iter=5000;
% Generate original matrix and rhs
[A, b]=get_covariance_matrix(N, tau, noise);
x0=zeros(N,1);
% Run CG on original system
[x_A, res_A, iter_A]=CG(A, b, x0, tol, max_iter);
rel_res_A=res_A/norm(b);
final_rel_res_A=norm(b - A*x_A)/norm(b);

% Build updated matrix A_new
M=chol(A);
A_new=M\(A/M');
A_new=(A_new+A_new')/2;%We have small imaginary components in the computed eigenvalues without this process
% Run CG on updated system
[x_new, res_new, iter_new] = CG(A_new, b, x0, tol, max_iter);
r0_new= norm(b - A_new*x0);
rel_res_new=res_new /norm(b);
final_rel_res_new=norm(b-A_new*x_new)/norm(b);

% Spectrum information
lambda_A=eig(A);
lambda_new=eig(A_new);
cond_A=cond(A);
cond_A_new=cond(A_new);

% Print summary
fprintf('Original matrix A:\n');
fprintf('  iterations        = %d\n', iter_A);
fprintf('  cond(A)           = %.6e\n', cond_A);
fprintf('  final rel residual= %.6e\n', final_rel_res_A);

fprintf('\nUpdated matrix A_new:\n');
fprintf('  iterations        = %d\n', iter_new);
fprintf('  cond(A_new)       = %.6e\n', cond_A_new);
fprintf('  final rel residual= %.6e\n', final_rel_res_new);

% Plot relative residual histories
figure;
semilogy(1:length(rel_res_A), rel_res_A, 'o-', 'LineWidth', 1.2, 'MarkerSize', 4);
hold on;
semilogy(1:length(rel_res_new), rel_res_new, 's-', 'LineWidth', 1.2, 'MarkerSize', 4);
grid on;
xlabel('Iteration');
ylabel('Relative residual norm');
title('CG convergence comparison');
legend('A', 'A_{new}', 'Location', 'northeast');

% Overlay eigenvalues for direct comparison
figure;
semilogy(sort(lambda_A, 'descend'), 'o-', 'LineWidth', 1.1, 'MarkerSize', 4);
hold on;
semilogy(sort(lambda_new, 'descend'), 'o-', 'LineWidth', 1.1, 'MarkerSize', 4);
grid on;
xlabel('Index');
ylabel('Eigenvalue (log scale)');
title('Spectrum comparison: A vs A_{new}');
legend('A', 'A_{new}', 'Location', 'best');
