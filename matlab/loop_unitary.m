1;
n = 8;
m = 4;
max_itr = 200;
tol=1e-10;
max_test = 1000000;
% load('/tmp/T_unitary_6_3.mat')
load('/tmp/x_init.mat');

[all_eig, n_runs] = ...
    find_all_complex_eigen_pairs(A, max_itr, max_test, tol);
res_name = sprintf('/tmp/res_%d_%d.mat', n, m);
res_x = all_eig.x;
res_lbd = all_eig.lbd;
res_is_real = all_eig.is_real;
res_is_self_conj = all_eig.is_self_conj;
save(res_name, 'res_x', 'res_lbd', 'res_is_real', 'res_is_self_conj', '-v7');
res = unitary_compare(n, m);
if 0
    n = 10;
    m = 4;
    res = unitary_compare(n, m);

    n = 15;
    m = 4;
    res = compare(n, m);

    n = 20;
    m = 3;
    res = compare(n, m);
    s1 = num2str(res(:,1), 6);
    s2 = num2str(res(:,5), 6);
    disp(size(unique(s1, 'rows')));
    disp(size(unique(s2, 'rows')));
    disp(size(setdiff(unique(s1, 'rows'), unique(s2, 'rows'), 'rows')));
    disp(size(setdiff(unique(s2, 'rows'), unique(s1, 'rows'), 'rows')));
end

function res = unitary_compare(n, m)
  nn = sprintf("/tmp/T_unitary_%d_%d.mat", n, m);
  disp(sprintf("Doing %s" , nn));
  load(nn);

  max_itr = 200;
  delta = 1e-10;

  n_test = 10;
  res = nan(n_test, 5+n);
  for i = 1:n_test
    x_init_r = randn(2*n,1);
    x_init = (x_init_r(1:n) + x_init_r(n+1:end)*1j) / norm(x_init_r);

    [x,lambda,ctr,run_time,converge,err] =  schur_form_rayleigh_unitary(T,max_itr,delta,x_init);
    res(i,1) = lambda;
    res(i,2) = ctr;
    res(i,3) = run_time;
    res(i,4) = converge;
    res(i,5) = err;
    res(i,6:end) = x;
  end
  disp (sprintf("mean schur %f ortho %f ", mean(res(1:end, 3)),
		mean(res(1:end, 3))));
  % disp(res(:,6:end));
  sfile = sprintf("/tmp/res_%d_%d.mat", n, m);
  save sfile, res;
end


