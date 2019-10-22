function [x,lambda,ctr,run_time,converge, err] = ...
     schur_form_rayleigh(T,max_itr,delta,x_init)

% Code by Du Nguyen
% ---------------------------------------------------------
% DESCRIPTION:
% 	This function implements the Rayleigh Quotient Iteration method
%   for computing real eigenpairs of symmetric tensors

%
% Input:    T - A cubic real and symmetric tensor
%           max_itr - maximum number of iterations until
%                     termination
%           delta - convergence threshold
%           x_init(opt) - initial point
%
% DEFAULT:
%   if nargin<4 x_init is chosen randomly over the unit sphere
%
% Output:   x - output eigenvector
%           ctr - number of iterations till convergence
%           run_time - time till convergence
%           convergence (1- converged)
% ---------------------------------------------------------
% FOR MORE DETAILS SEE:
% Lagrange Multipliers and Rayleigh Quotient Iteration
% Du Nguyen (2019)		    
% ---------------------------------------------------------

tic;

% get tensor dimensionality and order
n_vec = size(T);
m = length(n_vec);
n = size(T,1);
R = 1;
converge = 0;

% if not given as input, randomly initialize
if nargin<4
    x_init = randn(n,1);
    x_init = x_init/norm(x_init);
end

% init lambda_(k) and x_(k)

x_k = x_init;
T_x_m_2 = symmetric_tv_mode_product(T,x_k,m-2);
T_x_m_1 = T_x_m_2*x_k;
lambda = x_k' * T_x_m_1;
ctr = 1;

while(R>delta && ctr<max_itr)    
    rhs = cat(2, x_k, T_x_m_1);
    
    % compute Hessian H(x_k)
    H = (m-1)*T_x_m_2-lambda*eye(n);
    lhs = H\rhs;
    % fix eigenvector
    % y = -U_x_k*H_p_inv*U_x_k'*g;
    u = lhs(1:end, 1) * (sum(x_k .* lhs(1:end, 2)) / sum(x_k .* lhs(1:end, 1))) - lhs(1:end, 2);

    x_k_n = (x_k + u)/(norm(x_k + u));
    
    % update residual and lambda
    R = norm(x_k-x_k_n);
    x_k = x_k_n;
    % compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k)
    T_x_m_2 = symmetric_tv_mode_product(T,x_k,m-2);
    T_x_m_1 = T_x_m_2*x_k;
    lambda = x_k' * T_x_m_1;
    % lambda = symmetric_tv_mode_product(T,x_k,m);
        
    ctr = ctr+1;    
end

x = x_k;
run_time = toc;
err = norm(symmetric_tv_mode_product(T, x, m-1) - lambda * x);
if ctr<max_itr
    converge=1;
end
end
