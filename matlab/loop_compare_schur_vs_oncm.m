1;
load('test_tensors.mat');
save_res = cell(size(save_cell));
n_init = 300;
for i = 1:length(save_cell)
  ttype = save_cell{i, 1};
  m = ttype(1);
  n = ttype(2);
  tdata = save_cell{i, 2};
  tsize = size(tdata);
  save_res{i, 1} = ttype;
  ret = nan(tsize(1), n_init, 12);
  for jj = 1:tsize(1)
    T = reshape(tdata(jj, :), tsize(2:end));
    disp(sprintf("doing m=%d n=%d trial=%d", m, n, jj));
    ret(jj, :, :) = compare_run_time(n, m, T, n_init);
  end
  save_res{i, 2} = ret;
end
save('matlab_save_res.mat', 'save_res', '-v7');

function res = compare_run_time(n, m, T, n_init)
  % nn = sprintf("T_%d_%d.mat", n, m);
  % disp(sprintf("Doing %s" , nn));
  % load(nn);

  max_itr = 200;
  delta = 1e-10;


  res = nan(n_init, 12);
  for i = 1:n_init
    x_init = randn(n,1);
    x_init = x_init/norm(x_init);

    [x,lambda,ctr,run_time,converge] = orthogonal_newton_correction_method(T,max_itr,delta,x_init);
    [sr_x,sr_lambda,sr_ctr,sr_run_time,sr_converge] =  schur_form_rayleigh_raw(T,max_itr,delta,x_init);
    [s_x,s_lambda,s_ctr,s_run_time,s_converge] =  schur_form_rayleigh(T,max_itr,delta,x_init);
    res(i,1) = lambda;
    res(i,2) = ctr;
    res(i,3) = run_time;
    res(i,4) = converge;
    
    res(i,5) = sr_lambda;
    res(i,6) = sr_ctr;
    res(i,7) = sr_run_time;
    res(i,8) = sr_converge;

    res(i,9) = s_lambda;
    res(i,10) = s_ctr;
    res(i,11) = s_run_time;
    res(i,12) = s_converge;

  end
  disp (sprintf("time mean schur %f schur_raw %f ortho %f ",...
		mean(res(1:end, 11)), mean(res(1:end, 7)), mean(res(1:end, 3))));
  disp (sprintf("cnt mean schur %f schur_raw %f ortho %f ",...
		mean(res(1:end, 10)), mean(res(1:end, 6)),...
		mean(res(1:end, 2))));
  disp (sprintf("converge mean schur %f schur_raw %f ortho %f ",...
		mean(res(1:end, 12)), mean(res(1:end, 8)),...
		mean(res(1:end, 4))));

  disp (sprintf("mean diff raw %f", (1 - mean(res(1:end, 7)) ...
      ./ mean(res(1:end, 3)))));
  disp (sprintf("mean diff full schur %f", (1 - mean(res(1:end, 11)) ./ mean(res(1:end, 3)))));

  % sfile = sprintf("/tmp/res_%d_%d.mat", n, m);
  % save sfile, res;
end

% function unused()
% grid3 = cat(2, 3*ones(length(2:9), 1), (2:9)');
% grid4 = cat(2, 4*ones(length(2:8), 1), (2:8)');
% grid5 = cat(2, 5*ones(length(2:4), 1), (2:4)');
% grid6 = cat(2, 6*ones(length(2:4), 1), (2:4)');
% grids = cat(1, grid3, grid4, grid5, grid6);
% grids = grids(1:2, :);
% 1;
% end
% end
