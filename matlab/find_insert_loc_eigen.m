function [is_new, norm_lbd, good_x, is_self_conj, is_real] = ...
	 find_insert_loc_eigen(all_eig, x, lbd, eig_cnt, m, tol)
%
%    force eigen values to be positive if possible
%    if x is not similar to a vector in all_eig.x
%    then:
%       insert pair x, conj(x) if x is not self conjugate
%       otherwise insert x
%    all_eig has a structure: lbd, x, is_self_conj, is_real

  [is_self_conj, is_real, norm_lbd, norm_x] = ...
      normalize_real_positive(lbd, x, m, tol);
  if is_self_conj
    good_x = [norm_x];
  else
    good_x = [norm_x, conj(norm_x)];
  end
  factors = all_eig.x(1:eig_cnt, :) * conj(norm_x);
  fidx = find(abs(factors.^(m-2) - 1) < tol);
  if size(fidx, 1) > 0
    all_diffs = all_eig.x(fidx, 1:end) -...
                factors(fidx) * norm_x.';
    if size(find(sum(abs(all_diffs)) < tol * size(x,1)), 1) == 0
      is_new = 1;
      return;
    end
    is_new = 0;
    return;
  end
  is_new = 1;
end
