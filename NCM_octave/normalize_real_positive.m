function [is_self_conj,is_real,new_lbd,new_x] = ...
  normalize_real_positive (lbd,x,m,tol)
% First try to make it to a real pair
%  if not possible. If not then make lambda real
%  return is_self_conj, is_real, new_lbd, new_x

  u = conj(sqrt(x.' * x));
  is_self_conj = norm(conj(x) - u*u*x) < tol;
  new_x = x * u;

  if sum(imag(new_x)) < tol
  % try to flip. if u^(m-2) > 0 use it:
    lbd_factor = u^(m-2);
    if abs(lbd_factor*lbd_factor - 1) < tol
      lbd_factor = real(lbd_factor);
      if lbd * lbd_factor > 0
	is_real = 1;
	new_lbd = lbd * lbd_factor;
        return;
      elseif mod(m, 2) == 1
	is_real = 1;
	new_lbd = -lbd * lbd_factor;
	new_x = -new_x;
        return;
      else
	is_real = 1;
	new_lbd = lbd * lbd_factor;
        return;
      end
    end
  end
  if lbd < 0
    is_real = 0;
    new_lbd = -lbd;
    new_x = x * exp(pi/(m-2)*1j);
    return;
  else
    is_real = 0;
    new_lbd = lbd;
    new_x = x;
    return;
  end
end
