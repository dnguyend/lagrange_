function [x,lbd,ctr,run_time,converge,err] = ...
	 schur_form_rayleigh_unitary(T,max_itr,delta,x_init)
% Schur form rayleigh chebyshev unitary
% T and x are complex. Constraint is x^H x = 1
% lbd is real

% get tensor dimensionality and order
  tic;
  n_vec = size(T);
  m = length(n_vec);
  n = size(T, 1);
  R = 1;
  converge = 0;
% init lambda_(k) and x_(k)
  x_k = x_init;
  T_x_m_2 = symmetric_tv_mode_product(T,x_k,m-2);
  T_x_m_1 = T_x_m_2 * x_k;
    
  lbd = real(x_k' * T_x_m_1);
  ctr = 0;

  while((R > delta) && (ctr < max_itr))
    % compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
    rhs = cat(2, x_k, T_x_m_1);
       
    % compute Hessian H(x_k)
    H = (m-1)*T_x_m_2-lbd*eye(n);
    lhs = H\rhs;
    % fix eigenvector
    y = lhs(1:end, 1) * (real(x_k' * lhs(1:end, 2)) / real(x_k' * lhs(1:end, 1))) - lhs(1:end, 2);
    x_k_n = (x_k + y) / norm(x_k + y);
    %  update residual and lbd
    R = norm(x_k-x_k_n);
    x_k = x_k_n;
    T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2);
    T_x_m_1 = T_x_m_2 * x_k;

    lbd = real(x_k' * T_x_m_1);
    % print('ctr=%d lbd=%f' % (ctr, lbd))
    ctr = ctr + 1;
  end 
  x = x_k;
  run_time = toc;
  err = norm(symmetric_tv_mode_product(T, x, m-1) - lbd * x);
  if ctr < max_itr
    converge = 1;
  end
end
