% INPUTS:
%   A - n x n symmetric positive definite matrix
%   b- n x 1 right-hand side vector
%   x0 - n x 1 initial guess (use zeros(n,1) if unsure)
%   tol- convergence tolerance (e.g., 1e-6)
%   max_iter- maximum number of iterations
%
% OUTPUTS:
%   x- approximate solution vector
%   res_norm - vector of residual norms at each iteration
%   iter- number of iterations performed
function[x, res_norm, iter]=CG(A, b, x0, tol, max_iter)
%Initialisation
x=x0;
r=b-A*x0; % r_0=b - A*x_0
p=r;  % p_0=r_0
res_norm=zeros(max_iter, 1);
%Iteration loop
for j = 1 : max_iter
    Ap=A*p; % Compute A*p_j
    rr=r' * r; % (r_j, r_j)
    alpha=rr/(Ap' * p); % alpha_j = (r_j,r_j)/(A p_j, p_j)
    x=x+alpha * p; %x_{j+1} = x_j + alpha_j * p_j
    r=r-alpha * Ap; %r_{j+1} = r_j - alpha_j * A p_j
    res_norm(j)= norm(r);% Track residual norm
    %Convergence check
    if res_norm(j) < tol
        fprintf('Converged at iteration %d\n', j);
        iter = j;
        res_norm=res_norm(1:j);
        return;
    end
    beta=(r' * r)/rr; %beta_j = (r_{j+1},r_{j+1})/(r_j,r_j)
    p=r+beta * p; %p_{j+1} = r_{j+1} + beta_j * p_j
end
% If we exit without converging
fprintf('Warning: did not converge within %d iterations.\n', max_iter);
iter= max_iter;
res_norm= res_norm(1:max_iter);
end