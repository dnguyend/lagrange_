1;

load('test_tensors.mat');

n_tensor_trials = size(save_cell{1, 2}, 1);
save_res = cell(size(save_cell,1)*n_tensor_trials, 4);
max_test = 1000000;
max_itr = 200;
tol=1e-10;

res_idx = 1;
for i = 1:length(save_cell)
  ttype = save_cell{i, 1};
  m = ttype(1);
  n = ttype(2);
  tdata = save_cell{i, 2};
  tsize = size(tdata);

  for jj = 1:tsize(1)
    save_res{res_idx, 1} = [m, n, jj];
    T = reshape(tdata(jj, :), tsize(2:end));
    disp(sprintf("doing m=%d n=%d trial=%d", m, n, jj));
    [all_eig, n_runs, time_90, time_all] = ...
    find_all_complex_eigen_pairs(T, max_itr, max_test, tol);
    err = norm(symmetric_tv_mode_product(T, all_eig.x(1:1, :).', m-1)...
	       - all_eig.lbd(1)* all_eig.x(1:1, :).');
    if err > tol
      fprintf("problem with m=%d n-%d jj=%d err=%f", m, n, jj, err);
    end
    save_res{res_idx, 2} = all_eig;
    save_res{res_idx, 3} = time_90;
    save_res{res_idx, 4} = time_all;
    res_idx = res_idx + 1;
  end
end
save('matlab_save_unitary_res.mat', 'save_res', '-v7');