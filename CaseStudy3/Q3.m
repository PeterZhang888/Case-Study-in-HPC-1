clear; clc; close all;
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
% Store results 
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
