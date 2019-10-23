import numpy as np


def generate_symmetric_tensor(k, m):
    """Generating symmetric tensor size k,m
    """
    A = np.full(tuple(m*[k]), np.nan)
    current_idx = np.zeros(m, dtype=int)
    active_i = m - 1
    A[tuple(current_idx)] = np.random.rand()
    while True:
        if current_idx[active_i] < k - 1:
            current_idx[active_i] += 1
            if np.isnan(A[tuple(current_idx)]):
                i_s = tuple(sorted(current_idx))
                if np.isnan(A[i_s]):
                    A[i_s] = np.random.rand()
                    # print('Doing %s' % str(i_s))
                A[tuple(current_idx)] = A[i_s]
                # print('Doing %s' % str(current_idx))
        elif active_i == 0:
            break
        else:
            next_pos = np.where(current_idx[:active_i] < k-1)[0]
            if next_pos.shape[0] == 0:
                break
            current_idx[next_pos[-1]] += 1
            current_idx[next_pos[-1]+1:] = 0
                        
            active_i = m - 1
            if np.isnan(A[tuple(current_idx)]):
                i_s = tuple(sorted(current_idx))
                if np.isnan(A[i_s]):
                    A[i_s] = np.random.rand()
                    # print('Doing %s' % str(i_s))
                A[tuple(current_idx)] = A[i_s]
                # print('Doing %s' % str(current_idx))
    return A


def check_symmetric(T):
    def perm_map(j, ix):
        return [ix[j[a]] for a in range(len(ix))]
    
    from itertools import permutations
    m = len(T.shape)
    n = T.shape[0]
    for j in permutations(np.arange(m)):
        bad = False
        for i in range(1000):
            idx = np.random.randint(0, n, (m))
            new_idx = perm_map(j, idx)
            if T[tuple(new_idx)] != T[tuple(idx)]:
                print(j)
                print(idx)
                print(new_idx)
                bad = True
                break
        if bad:
            break
    if not bad:
        print("all good")


def generate_all_tensors_for_matlab(fname):
    from scipy.io import savemat
    from numpy import concatenate, ones, nan
    np.random.seed(1)
    n_trials = 20
    grid_dict = {3: np.array(range(2, 10), dtype=int),
                 4: np.array(range(2, 9), dtype=int),
                 5: np.array(range(2, 5), dtype=int),
                 6: np.array(range(2, 5), dtype=int)}
    grid_list = concatenate([concatenate(
        [a*ones((len(grid_dict[a]), 1), dtype=int),
         grid_dict[a].reshape(-1, 1)],
        axis=1) for a in sorted(grid_dict.keys())], axis=0)

    save_cell = np.empty((len(grid_list), 2), dtype=object)
    for i in range(len(grid_list)):
        m, n = tuple(grid_list[i, :])
        print(m, n)
        Tc = np.full([n_trials] + m*[n], nan, dtype=float)
        for j in range(n_trials):
            Tc[j, :] = generate_symmetric_tensor(n, m)
        save_cell[i, 0] = grid_list[i, :]
        save_cell[i, 1] = Tc
    savemat(fname, {'save_cell': save_cell})

    
def check_results_real_pairs():
    from scipy.io import loadmat
    # res = loadmat('./matlab/octave_save_res.mat')
    res = loadmat('./matlab/matlab_save_res.mat')
    n_scenarios = res['save_res'].shape[0]
    save_mean = np.zeros((n_scenarios, 3))
    save_mean_not_2 = np.zeros((n_scenarios, 3))
    for j in range(n_scenarios):
        # res['save_res'][j, 1]
        # has shape # trial tensors (20)
        # n_initials
        # 12 for results: 0, 4, 8: lambda
        # 1, 5, 9: iteration counts
        # 2, 6, 10: time
        # 3, 7, 11: convergence
        # so we take average over the first two dimensions

        tot_time = res['save_res'][j, 1][:, :, 2+4*np.arange(3)].mean(
            axis=(0, 1))
        tot_time_save = 1 - tot_time / tot_time[0]
        
        print("mean_time m=%d n=%d orth=%f schur_raw=%f schur=%f ::: schur_raw_save=%f schur_save=%f" %
              tuple(list(res['save_res'][j, 0][0, :]) +
                    list(res['save_res'][j, 1][:, :, 2+4*np.arange(3)].mean(
                        axis=(0, 1))) + list(tot_time_save[1:])))
        
        save_mean[j, :] += res['save_res'][j, 1][:, :, 2+4*np.arange(3)].mean(
            axis=(0, 1))
        if res['save_res'][j, 0][0, 1] > 2:
            save_mean_not_2[j, :] += res['save_res'][j, 1][:, :, 2+4*np.arange(3)].mean(
            axis=(0, 1))

        
        print("mean_lbd m=%d n=%d orth=%f schur_raw=%f schur=%f" %
              tuple(list(res['save_res'][j, 0][0, :]) +
                    list(res['save_res'][j, 1][:, :, 4*np.arange(3)].mean(
                        axis=(0, 1)))))
        print("mean_itr m=%d n=%d orth=%f schur_raw=%f schur=%f" %
              tuple(list(res['save_res'][j, 0][0, :]) +
                    list(res['save_res'][j, 1][:, :, 1+4*np.arange(3)].mean(
                        axis=(0, 1)))))

        print("mean_conv  m=%d n=%d orth=%f schur_raw=%f schur=%f" %
              tuple(list(res['save_res'][j, 0][0, :]) +
                    list(res['save_res'][j, 1][:, :, 3+4*np.arange(3)].mean(
                        axis=(0, 1)))))

    print("total mean orth=%f schur_raw=%f schur=%f" %
          tuple(save_mean.mean(axis=0)))
    improve = save_mean.mean(axis=0)
    print('improve raw=%f schur=%f' % tuple(1 - improve[1:] / improve[0]))
    print("for n> 2: total mean orth=%f schur_raw=%f schur=%f" %
          tuple(save_mean_not_2.mean(axis=0)))
    improve = save_mean_not_2.mean(axis=0)
    print('improve raw=%f schur=%f' % tuple(1 - improve[1:] / improve[0]))


def check_results_complex_pairs():
    from scipy.io import loadmat
    import pandas as pd
    # res = loadmat('./matlab/octave_save_res.mat')
    res = loadmat('./matlab/matlab_save_unitary_res.mat')
    n_scenarios = res['save_res'].shape[0]
    # save_mean = np.zeros((n_scenarios, 3))
    # aggregate to table:
    # (m, n, j, # eigen values, #run time 90%, run time 100%
    # then run some pandas group by statements
    sum_table = pd.DataFrame(
        {'m': np.zeros((n_scenarios), dtype=int),
         'n': np.zeros((n_scenarios), dtype=int),
         'n_trys': np.zeros((n_scenarios), dtype=int),
         'n_pairs': np.zeros((n_scenarios), dtype=int),
         'n_real_pairs': np.zeros((n_scenarios), dtype=int),
         'n_self_conj_pairs': np.zeros((n_scenarios), dtype=int),
         'n_multiple_eigen': np.zeros((n_scenarios), dtype=int),
         'time_90': np.zeros((n_scenarios), dtype=float),
         'time_all': np.zeros((n_scenarios), dtype=float)},
        columns=['m', 'n', 'n_trys', 'n_pairs', 'n_real_pairs',
                 'n_self_conj_pairs', 'n_multiple_eigen',
                 'time_90', 'time_all'])
                             
    for j in range(n_scenarios):
        m = res['save_res'][j, 0][0, 0]
        n = res['save_res'][j, 0][0, 1]
        n_trys = res['save_res'][j, 0][0, 2]

        sum_table.loc[j, 'm'] = m
        sum_table.loc[j, 'n'] = n
        sum_table.loc[j, 'n_trys'] = n_trys
        eig_cell = res['save_res'][j, 1][0]

        dtypes = eig_cell.dtype
        dtypes_dict = dict((dtypes.names[a], a)
                           for a in range(len(dtypes.names)))

        n_pairs = res['save_res'][j, 1][0][0][dtypes_dict['lbd']].shape[0]
        sum_table.loc[j, 'n_pairs'] = n_pairs

        n_real_pairs = np.sum(
            res['save_res'][j, 1][0][0][dtypes_dict['is_real']])
        sum_table.loc[j, 'n_real_pairs'] = n_real_pairs

        n_self_conj_pairs = np.sum(
            res['save_res'][j, 1][0][0][dtypes_dict['is_self_conj']])
        sum_table.loc[j, 'n_self_conj_pairs'] = n_self_conj_pairs

        # find multiple eigen:
        # typically one lbd has one or two eigen vectors.
        # some cases we have multiple eigen vectors
        # we print them out here

        u, cnt = np.unique(['%.6f' % np.abs(a) for a in
                            res['save_res'][j, 1][0][0][dtypes_dict['lbd']]],
                           return_counts=True)
        dup_cnt = [(u[aa], cnt[aa]) for aa in range(len(u)) if cnt[aa] > 2]
        if len(dup_cnt) > 0:
            print('m=%d n=%d j=%d dup=%s' % (m, n, j, str(dup_cnt)))
            sum_table.loc[j, 'n_multiple_eigen'] = np.sum(
                [a[1] for a in dup_cnt])
        sum_table.loc[j, 'time_90'] = res['save_res'][j, 2][0, 0]
        sum_table.loc[j, 'time_all'] = res['save_res'][j, 3][0, 0]

    # sum_by_m_n = sum_table.groupby(['m', 'n']).sum()
    mean_by_m_n = sum_table.groupby(['m', 'n']).mean()
    mean_by_m_n.n_trys = sum_table[['m', 'n', 'n_trys']].groupby(
        ['m', 'n']).count()
    mean_by_m_n.n_pairs = sum_table[['m', 'n', 'n_pairs']].groupby(
        ['m', 'n']).mean()

    mean_by_m_n.n_real_pairs = sum_table[['m', 'n', 'n_real_pairs']].groupby(
        ['m', 'n']).mean()
    mean_by_m_n.n_multiple_eigen = sum_table[['m', 'n', 'n_multiple_eigen']].groupby(
        ['m', 'n']).mean()
    mean_by_m_n.n_multiple_eigen /= mean_by_m_n.n_pairs
    mean_by_m_n.time_90 /= mean_by_m_n.n_pairs
    mean_by_m_n.time_all /= mean_by_m_n.n_pairs


    # from IPython.core.display import display, HTML
    with open('/tmp/sum.html', 'w') as hf:
        hf.write("%s\n" % mean_by_m_n.to_html())
    with open('/tmp/mean.tex', 'w') as hf:
        hf.write("%s\n" % mean_by_m_n.to_latex())
        
    with open('/tmp/sum_detail.html', 'w') as hf:
        hf.write("%s\n" % sum_table.to_html())       
        
            

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        generate_all_tensors_for_matlab(sys.argv[1])
    else:
        print("save tensors to test_tensors.mat")
        generate_all_tensors_for_matlab("test_tensors.mat")
