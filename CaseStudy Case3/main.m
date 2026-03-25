% Q2
% Test the Conjugate Gradient algorithm on covariance
% matrices generated from the Matern kernel.
%
% tolerance = 10e-8
% tau = 20
% noise = 0.005
% N = 1024
% Chosen additional sizes: 512 and 2048.
clear; clc; close all;
% Parameters
 tol= 10e-8;
 tau= 20;
 noise= 0.005;
 dims=[512, 1024, 2048];
% Maximum iterations: CG for SPD matrices converges in at most N steps in exact arithmetic.
results=struct();
for k= 1:length(dims)
    N=dims(k);
    fprintf('N = %d\n', N);
    % Generate matrix and rhs
    [A,b]=get_covariance_matrix(N, tau, noise);
    % Initial guess
    x0= zeros(N,1);
    % Set maximum iteration as the matrix size
    max_iter=N;
    % Run CG
    [x, res_norm, iter]=CG(A, b, x0, tol, max_iter);
    % Relative residual norm history
    rel_res=res_norm/norm(b);
    % Final relative residual
    final_rel_res=norm(b-A*x)/norm(b);
    % Store results
    results(k).N=N;
    results(k).A=A;
    results(k).b=b;
    results(k).x=x;
    results(k).iter= iter;
    results(k).res_norm=res_norm;
    results(k).rel_res=rel_res;
    results(k).final_rel_res=final_rel_res;
    % Print summary
    fprintf('  Final relative residual: %.6e\n', final_rel_res);
end

% Plot relative residual norms versus iteration
figure;
hold on;
for k = 1:length(results)
    semilogy(1:length(results(k).rel_res), results(k).rel_res, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('N = %d', results(k).N));
end
xlabel('Iteration');
ylabel('Relative residual norm');
title('CG convergence for Matern covariance matrices');
legend('Location', 'best');
grid on;
hold off;








% Q3
N=1024;
tau=416;
tol=1e-8;    
max_iter=100000;
x0=zeros(N,1);
noise_vals=[0.5, 0.05, 0.005, 5e-4, 5e-5, 5e-6, 5e-7, 5e-8, 0.5e-8];
num_cases=length(noise_vals);
iterations=zeros(num_cases,1);
cond_nums=zeros(num_cases,1);
for k = 1:num_cases
    noise=noise_vals(k);
    [A,b]=get_covariance_matrix(N, tau, noise);
    %Condition number of SPD matrix
    cond_nums(k) = cond(A);
    %Run CG
    [x, res_norm, iter]=CG(A, b, x0, tol, max_iter);
    iterations(k)=iter;
end
% Store results in a table for display
results_table = table(noise_vals(:), iterations, cond_nums, ...
    'VariableNames', {'Noise','Iterations','ConditionNumber'});
disp(results_table);

% Plot iterations against noise
figure;
semilogx(noise_vals, iterations, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('Noise parameter');
ylabel('Iterations to convergence');
title('CG iterations vs noise level');
set(gca, 'XDir', 'reverse');

% Plot condition number against noise
figure;
loglog(noise_vals, cond_nums, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('Noise parameter');
ylabel('cond(A)');
title('Condition number versus noise level');
set(gca, 'XDir', 'reverse');






% Q4
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
