1;
n = 8;
m = 4;
res = compare(n, m);
if 1
    n = 10;
    m = 4;
    res = compare(n, m);

    n = 15;
    m = 4;
    res = compare(n, m);

    n = 20;
    m = 3;
    res = compare(n, m);
end
s1 = num2str(res(:,1), 6);
s2 = num2str(res(:,5), 6);
disp(size(unique(s1, 'rows')));
disp(size(unique(s2, 'rows')));
disp(size(setdiff(unique(s1, 'rows'), unique(s2, 'rows'), 'rows')));
disp(size(setdiff(unique(s2, 'rows'), unique(s1, 'rows'), 'rows')));

function res = compare(n, m)
  nn = sprintf("/tmp/T_%d_%d.mat", n, m);
  disp(sprintf("Doing %s" , nn));
  load(nn);

  max_itr = 200;
  delta = 1e-10;

  n_test = 300;
  res = nan(n_test, 8);
  for i = 1:n_test
    x_init = randn(n,1);
    x_init = x_init/norm(x_init);

    [x,lambda,ctr,run_time,converge] = orthogonal_newton_correction_method(T,max_itr,delta,x_init);

    [s_x,s_lambda,s_ctr,s_run_time,s_converge] =  schur_form_rayleigh(T,max_itr,delta,x_init);
    res(i,1) = lambda;
    res(i,2) = ctr;
    res(i,3) = run_time;
    res(i,4) = converge;
    
    res(i,5) = s_lambda;
    res(i,6) = s_ctr;
    res(i,7) = s_run_time;
    res(i,8) = s_converge;
  end
  disp (sprintf("mean schur %f ortho %f ", mean(res(1:end, 7)),  mean(res(1:end, 3))));
  disp (sprintf("mean diff %f", (1 - mean(res(1:end, 7)) ./ mean(res(1:end, 3)))));
  sfile = sprintf("/tmp/res_%d_%d.mat", n, m);
  save sfile, res;
end


