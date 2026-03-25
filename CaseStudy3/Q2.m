% Test the Conjugate Gradient algorithm on covariance matrices generated from the Matern kernel.
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
% Maximum iterations: CG for SPD matrices converges in at most N steps.
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

