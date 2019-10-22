function [n_eig] = complex_eigen_cnt(n, m)
% Implement the Cartwright Sturmfels formula
  if m == 2     
    n_eig = n;
  else
    n_eig = ((m-1)^n-1) / (m-2);
  end
  return;
end
