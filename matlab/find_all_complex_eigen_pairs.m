function [all_eig, n_runs, time_90, time_all] = ...
	 find_all_complex_eigen_pairs(...
	     A, max_itr, max_test, tol)
% output is the table of results
% 2n*+2 columns: lbd, is self conjugate, x_real, x_imag
% This is the raw version, since the output vector x
% is not yet normalized to be real when possible
  time_start_find = tic;
  n = size(A,1);
  m = length(size(A));
  n_eig = complex_eigen_cnt(n, m);
  n_90 = 0.9 * n_eig;
  all_eig = struct('lbd', nan(n_eig,1));
  all_eig.x =  complex(nan(n_eig, n));
  all_eig.is_self_conj = zeros(n_eig,1);
  all_eig.is_real = zeros(n_eig,1);
  eig_cnt = 0;
  found_90 = 0;
  for jj = 1:max_test
    x0r = randn(2*n,1);
    x0r = x0r/norm(x0r);
    x0 = x0r(1:n) + x0r(n+1:end) * 1.j;
    % if there are odd numbers left,
    % try to find a real root
    draw = unifrnd(0, 1);
    % 50% try real root
    if (draw < .5) && (mod(n_eig - eig_cnt, 2) == 1)
      try
        [x_r, lbd, ctr, run_time, converge, err] = schur_form_rayleigh(A, max_itr, tol, real(x0));
        x = x_r + 1j * zeros(size(x_r));
      catch ME          
        disp(getReport(ME));
        continue;
      end
    else
      try
        [x, lbd, ctr, run_time, converge, err] = ...
            schur_form_rayleigh_unitary(A, max_itr, tol, x0);
      catch ME
        disp(getReport(ME));
        continue;
      end
     end
    old_eig = eig_cnt;
    if converge && (err < tol)
      [is_new, norm_lbd, good_x, is_self_conj, is_real] = ...
      find_insert_loc_eigen(all_eig,x,lbd,eig_cnt,m,tol);
      if is_new
	for j = 1:size(good_x,2)
	  all_eig.lbd(eig_cnt+j) = norm_lbd;
	  all_eig.x(eig_cnt+j, :) = good_x(:, j);
	  all_eig.is_self_conj(eig_cnt+j) = is_self_conj;
          all_eig.is_real(eig_cnt+j) = is_real;
	end
	eig_cnt = eig_cnt + size(good_x,2);
      end
      if (found_90 == 0) && (eig_cnt > n_90)
	found_90 = 1;
	time_90 = toc(time_start_find);
      end
	
      if eig_cnt >= n_eig
        break;
      elseif (eig_cnt > old_eig) && (mod(eig_cnt, 10) == 0)
        disp(sprintf('Found %d eigenpairs', eig_cnt));
        % fflush(stdout);
      end
    end
  end
  time_all = toc(time_start_find);  
  n_runs = jj;
  return;
end
		      
