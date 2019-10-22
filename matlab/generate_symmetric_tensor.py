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


if __name__ == '__main__':
    from scipy.io import savemat
    k = 10
    m = 4
    T = generate_symmetric_tensor(k, m)
    savemat('/tmp/T_%d_%d.mat' % (k, m), {'T': T})
