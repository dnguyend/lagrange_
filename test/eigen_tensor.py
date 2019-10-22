from __future__ import print_function
import numpy as np
import pandas as pd
import time

import lagrange_rayleigh.core.utils as utils
from lagrange_rayleigh.core.vector_lagrangian import explicit_vector_lagrangian
from lagrange_rayleigh.core.constraints import base_constraints

from lagrange_rayleigh.core.solver import rayleigh_quotient_iteration
# from lagrange_rayleigh.core.solver import rayleigh_chebyshev
from lagrange_rayleigh.core.eigen_tensor_solver import\
    orthogonal_newton_correction_method, schur_form_rayleigh,\
    symmetric_tv_mode_product, schur_form_rayleigh_linear,\
    schur_form_rayleigh_chebyshev
from lagrange_rayleigh.core.eigen_tensor_solver import\
        schur_form_rayleigh_chebyshev_unitary,\
        find_all_unitary_eigenpair


class eigen_tensor_lagrange(explicit_vector_lagrangian):
    def calc_H(self, x):
        return x[:, None]

    def F(self, x):
        v = self._args['A'].copy()
        for i in range(self._m-3):
            v = np.tensordot(v, x, axes=1)
        self._F2 = v
        self._F1 = np.tensordot(self._F2, x, axes=1)
        self._F0 = np.tensordot(self._F1, x, axes=1)
        self._F1 *= (self._m - 1)
        self._F2 *= (self._m - 2) * (self._m - 1)
        return self._F0
    
    def __init__(self, A):
        self._args = {'A': A}
        self._k = A.shape[0]
        self._m = len(A.shape)
        self._shape_in = (A.shape[0], 0)
        self._shape_out = (A.shape[0], 0)

    def calc_J_F(self, x):
        return self._F1

    def calc_J_H(self, x):
        return np.eye(x.shape[0]).reshape(
            x.shape[0], 1, x.shape[0])

    def J_F2(self, d_x):
        return np.tensordot(
            np.tensordot(self._F2, d_x, axes=1), d_x, axes=1)

    def J_C(self, d_x):
        return np.dot(
            self._state['J_C'], d_x)

    def calc_J_F2(self, x):
        return self._F2

    def calc_J_H2(self, x):
        pass

    def J_H2(self, d_x, d_lbd):
        return np.zeros((d_x.shape[0]))

    def J_C2(self, d_x):
        return 2 * np.dot(d_x.T, d_x).reshape(1)

    def calc_J_RAYLEIGH(self, x):
        return (- 2 * self['RAYLEIGH'] * x.T +
                self._F0.T + np.dot(x.T, self._F1)).reshape(1, -1)


def _test_eigen_tensor(k, m, max_err, max_itr, n_test):
    def sphere_func(x):
        return np.dot(x.T, x) - 1

    def sphere_jacobian(x):
        return 2 * x.reshape(1, -1)

    def sphere_retraction(x, u):
        return (x + u) / np.linalg.norm(x + u)

    sphere = base_constraints(
        shape_in=(k,),
        shape_constraint=(1,),
        equality=sphere_func)

    sphere.set_analytics(
        J_C=sphere_jacobian,
        retraction=sphere_retraction)

    A = utils.generate_symmetric_tensor(k, m)
    e = eigen_tensor_lagrange(A)
    e.constraints = sphere

    o_ncm_cnt = np.zeros(n_test, dtype=int)
    schur_cnt = np.zeros(n_test, dtype=int)
    ray_cnt = np.zeros(n_test, dtype=int)
    schur_cheb_cnt = np.zeros(n_test, dtype=int)

    o_ncm_err = np.zeros(n_test)
    schur_err = np.zeros(n_test)
    ray_err = np.zeros(n_test)
    schur_cheb_err = np.zeros(n_test)

    o_ncm_lbd = np.zeros(n_test)
    schur_lbd = np.zeros(n_test)
    ray_lbd = np.zeros(n_test)
    schur_cheb_lbd = np.zeros(n_test)

    o_ncm_time = np.zeros(n_test)
    schur_time = np.zeros(n_test)
    ray_time = np.zeros(n_test)
    schur_cheb_time = np.zeros(n_test)

    for jj in range(n_test):
        x0 = np.random.randn(k)
        x0 = x0 / np.linalg.norm(x0)

        # do orthogonal
        t_start = time.time()
        o_x, o_lbd, o_ctr, converge = orthogonal_newton_correction_method(
            A, max_itr, max_err, x_init=x0)
        t_end = time.time()
        o_ncm_cnt[jj] = o_ctr
        o_ncm_lbd[jj] = o_lbd
        o_ncm_err[jj] = np.linalg.norm(
            symmetric_tv_mode_product(
                A, o_x, m-1) - o_lbd * o_x)
        o_ncm_time[jj] = t_end - t_start

        # do schur_form_rayleigh
        t_start = time.time()
        if False:
            s_x, s_lbd, ctr, converge = schur_form_rayleigh(
                A, max_itr, max_err, x_init=x0)
        else:
            # s_x, s_lbd, ctr, converge = schur_form_rayleigh_chebyshev_linear(
            # A, max_itr, max_err, x_init=x0, do_chebyshev=True)
            s_x, s_lbd, ctr, converge, err = schur_form_rayleigh_chebyshev(
                A, max_itr, max_err, x_init=x0, do_chebyshev=False)

        t_end = time.time()
        schur_cnt[jj] = ctr
        schur_lbd[jj] = s_lbd
        schur_err[jj] = np.linalg.norm(
            symmetric_tv_mode_product(
                A, s_x, m-1) - s_lbd * s_x)
        schur_time[jj] = t_end - t_start

        # now do rayleigh

        t_start = time.time()
        res_ray = rayleigh_quotient_iteration(
            e, x0, max_err=max_err,
            max_iter=max_itr, verbose=False,
            exit_by_diff=True)
        t_end = time.time()
        ray_time[jj] = t_end - t_start
        ray_cnt[jj] = res_ray['n_iter']
        ray_lbd[jj] = res_ray['lbd']
        ray_err[jj] = np.linalg.norm(res_ray['err'])
        # print("doing rayleigh")
        # print(res_ray)
        # print(e.L(res_ray['x'], res_ray['lbd']))

        # now do rayleigh chebyshev
        t_start = time.time()
        if True:
            sch_x, sch_lbd, ctr, converge, err = schur_form_rayleigh_chebyshev(
                A, max_itr, max_err, x_init=x0, do_chebyshev=True)
        else:
            sch_x, sch_lbd, ctr, converge = schur_form_rayleigh_linear(
                A, max_itr, max_err, x_init=x0, u=None)
        t_end = time.time()
        """
        res_ray_cheb = rayleigh_chebyshev(
            e, x0, max_err=max_err, max_iter=max_itr,
            verbose=False, exit_by_diff=True)
        """

        schur_cheb_time[jj] = t_end - t_start
        schur_cheb_cnt[jj] = ctr
        schur_cheb_lbd[jj] = sch_lbd
        schur_cheb_err[jj] = np.linalg.norm(
            symmetric_tv_mode_product(
                A, sch_x, m-1) - sch_lbd * sch_x)
        schur_cheb_time[jj] = t_end - t_start

        # print("doing raychev")
        # print(res_ray_cheb)
        # print(e.L(res_ray_cheb['x'], res_ray_cheb['lbd']))

    summ = pd.DataFrame(
        {
            'o_ncm_iter': o_ncm_cnt,
            'schur_iter': schur_cnt,
            'ray_iter': ray_cnt, 'schur_cheb_iter': schur_cheb_cnt,
            'o_ncm_err': o_ncm_err,
            'schur_err': schur_err,
            'ray_err': ray_err, 'schur_cheb_err': schur_cheb_err,
            'o_ncm_lbd': o_ncm_lbd,
            'schur_lbd': schur_lbd,
            'ray_lbd': ray_lbd,
            'schur_cheb_lbd': schur_cheb_lbd,
            'o_ncm_time': o_ncm_time,
            'schur_time': schur_time,
            'ray_time': ray_time,
            'schur_cheb_time': schur_cheb_time
        },
        columns=['o_ncm_iter', 'o_ncm_lbd', 'o_ncm_err', 'o_ncm_time',
                 'schur_iter', 'schur_lbd', 'schur_err', 'schur_time',
                 'ray_iter', 'ray_lbd', 'ray_err', 'ray_time',
                 'schur_cheb_iter', 'schur_cheb_lbd', 'schur_cheb_err',
                 'schur_cheb_time'])
    return summ


def test_tensor_unitary_eigenpair():
    # output is the table of results
    # 2n*+2 columns: lbd, is real, real, complex eigenvalue
    from lagrange_rayleigh.core.eigen_tensor_solver import\
        schur_form_rayleigh_chebyshev_unitary

    n = 8
    m = 3
    tol = 1e-10
    max_itr = 200
    n_test = 10000

    A = utils.generate_symmetric_tensor(n, m)

    # n_eig = complex_eigen_cnt(n, m)

    su_x = np.zeros((n_test, n), dtype=np.complex)
    su_cnt = np.zeros(n_test, dtype=int)
    su_err = np.zeros(n_test)
    su_lbd = np.zeros(n_test)
    su_time = np.zeros(n_test)

    su_cheb_x = np.zeros((n_test, n), dtype=np.complex)
    su_cheb_cnt = np.zeros(n_test, dtype=int)
    su_cheb_err = np.zeros(n_test)
    su_cheb_lbd = np.zeros(n_test)
    su_cheb_time = np.zeros(n_test)

    for jj in range(n_test):
        x0r = np.random.randn(2*n)
        x0r /= np.linalg.norm(x0r)
        x0 = x0r[:n] + x0r[n:] * 1.j
        t_start = time.time()
        x, lbd, ctr, converge, err = schur_form_rayleigh_chebyshev_unitary(
            A, max_itr, tol, x_init=x0, do_chebyshev=False)
        
        t_end = time.time()
        # su_err[jj] = np.linalg.norm(
        # symmetric_tv_mode_product(
        # A, x, m-1) - lbd * x)
        su_err[jj] = err
        su_x[jj] = x
        su_cnt[jj] = ctr
        su_lbd[jj] = lbd
        su_time[jj] = t_end - t_start
        
        t_start = time.time()
        sc_x, sc_lbd, sc_ctr, converge, sc_err =\
            schur_form_rayleigh_chebyshev_unitary(
                A, max_itr, tol, x_init=x0, do_chebyshev=True)
        t_end = time.time()
        su_cheb_err[jj] = sc_err
        su_cheb_x[jj] = sc_x
        su_cheb_cnt[jj] = sc_ctr
        su_cheb_lbd[jj] = sc_lbd
        su_cheb_time[jj] = t_end - t_start

    summ = pd.DataFrame(
        {
            'su_time': su_time,
            'su_cnt': su_cnt,
            'su_lbd': su_lbd,
            'su_err': su_err,

            'su_cheb_time': su_cheb_time,
            'su_cheb_cnt': su_cheb_cnt,
            'su_cheb_lbd': su_cheb_lbd,
            'su_cheb_err': su_cheb_err

        },
        columns=['su_time', 'su_cnt', 'su_lbd', 'su_err',
                 'su_cheb_time', 'su_cheb_cnt', 'su_cheb_lbd', 'su_cheb_err'])
    # print(summ)
    print(summ[[a for a in summ.columns if 'err' in a]].describe())
    print(summ[[a for a in summ.columns if 'cnt' in a]].describe())
    print(summ[[a for a in summ.columns if 'lbd' in a]].describe())
    print(summ[[a for a in summ.columns if 'time' in a]].describe())


def check_eigen(all_eig, A, tol):
    m = len(A.shape)
    for i in range(all_eig.lbd.shape[0]):
        # first check eigen works
        err = np.sum(np.abs(
            symmetric_tv_mode_product(
                A, all_eig.x[i], m-1) - all_eig.lbd[i] * all_eig.x[i]))
        if err > tol:
            print("bad entry i=%d lbd=%f z=%s" % (
                i, all_eig.lbd[i], str(all_eig.x[i])))
    neg_factor = np.exp(np.pi/(m-2)*1j)
    neg_factors = np.power(neg_factor, np.arange(m-2))
    # second check real

    for i in range(all_eig.x.shape[0]):
        err = np.sum(np.abs(
            symmetric_tv_mode_product(
                A, all_eig.x[i].real, m-1) -
            all_eig.lbd[i] * all_eig.x[i].real))
        if (err < tol) != (all_eig.is_real[i]):
            print("bad real i=%d lbd=%f z=%s" % (
                i, all_eig.lbd[i], str(all_eig.x[i])))
    # number of real eigen pairs:
    print("number of real=%d total=%d " % (np.where(
        all_eig.is_real)[0].shape[0],
        all_eig.x.shape[0]))

    for i in range(all_eig.x.shape[0]):
        # third check no duplicate
        match_lbd = np.where(
            np.abs(np.abs(
                all_eig.lbd[:]) - np.abs(all_eig.lbd[i])) < tol)[0]
        match_other = match_lbd[match_lbd != i]
        if match_other.shape[0] == 0:
            continue

        for jj in range(m-2):
            dup = np.where(np.abs(
                all_eig.x[match_other] -
                all_eig.x[i] * neg_factors[jj]) < tol)[0]
            if dup.shape[0] > 0:
                print("bad dup i=%d lbd=%f z=%s, jj=%d" % (
                    i, all_eig.lbd[i], str(all_eig.x[i]), jj))
    # check duplicate again:
    rnk = np.vectorize(lambda a: '%.6f' % a)(all_eig.lbd)
    u, cnts = np.unique(rnk, return_counts=True)
    for iu in range(len(u)):
        if cnts[iu] > 2:
            mtc = np.where(rnk == u[iu])[0]
            print(mtc)
            print(all_eig.x[mtc])
            print(all_eig.lbd[mtc])


def test_find_all_unitary():
    n = 8
    m = 3
    tol = 1e-10
    max_itr = 200

    A = utils.generate_symmetric_tensor(n, m)

    # find from begining
    t_start = time.time()
    all_eig, n_runs = find_all_unitary_eigenpair(
        all_eig=None, eig_cnt=None, A=A, max_itr=max_itr, max_test=10, tol=tol)

    # continue finding more pairs
    all_eig, n_runs = find_all_unitary_eigenpair(
        all_eig, eig_cnt=None, A=A, max_itr=max_itr,
        max_test=int(1e6), tol=tol)

    t_end = time.time()
    tot_time = t_end - t_start
    print('tot time %f avg=%f' % (tot_time, tot_time / all_eig.x.shape[0]))

    # all_eig = fix_imag_eigen(all_eig_raw, m, tol)
    np.savez_compressed('/tmp/save_eigen_%d_%d.npz' % (
        n, m), A=A, lbd=all_eig.lbd,
                        x=all_eig.x, is_real=all_eig.is_real,
                        is_self_conj=all_eig.is_self_conj)


def detailed_all_unitary_eigenpair(A, max_itr, max_test=1e6, tol=1e-10):
    # output is the table of results
    # 2n*+2 columns: lbd, is real, real, complex eigenvalue
    n = A.shape[0]
    m = len(A.shape)
    # n_eig = complex_eigen_cnt(n, m)

    su_x = np.zeros((n_test, n), dtype=np.complex)
    su_cnt = np.zeros(n_test, dtype=int)
    su_err = np.zeros(n_test)
    su_lbd = np.zeros(n_test)
    su_time = np.zeros(n_test)

    for jj in range(n_test):
        x0r = np.random.randn(2*n)
        x0r /= np.linalg.norm(x0r)
        x0 = x0r[:n] + x0r[n:] * 1.j
        t_start = time.time()
        x, lbd, ctr, converge = schur_form_rayleigh_chebyshev_unitary(
            A, max_itr, tol, x_init=x0, do_chebyshev=False)

        su_err[jj] = np.linalg.norm(
            symmetric_tv_mode_product(
                A, x, m-1) - lbd * x)

        t_end = time.time()
        su_x[jj] = x
        su_cnt[jj] = ctr
        su_lbd[jj] = lbd
        su_time[jj] = t_end - t_start
    summ = pd.DataFrame(
        {
            'su_time': su_time,
            'su_cnt': su_cnt,
            'su_lbd': su_lbd,
            'su_err': su_err},
        columns=['su_time', 'su_cnt', 'su_lbd', 'su_err'])
    print(summ)
    return summ
    

def debug_matlab(A):
    import scipy.io as sio
    max_itr = 200
    tol = 1e-10
    # mm1 = sio.loadmat('/tmp/T_unitary_8_4.mat')
    # A = mm1['A']
    n = 8
    m = 4
    tol = 1e-10
    max_itr = 200

    A = utils.generate_symmetric_tensor(n, m)

    n = A.shape[0]
    x0r = np.random.randn(2*n)
    x0r /= np.linalg.norm(x0r)
    x_init = x0r[:n] + x0r[n:] * 1.j
    sio.savemat('/tmp/x_init.mat', {'x_init': x_init, 'A': A})
    """
    x, lbd, ctr, converge, err = schur_form_rayleigh_chebyshev_unitary(
        A, max_itr, tol, x_init=x_init, do_chebyshev=False)
    """
    res = sio.loadmat('/tmp/res_%d_%d.mat' % (n, m))
    for jj in range(res['res_x'].shape[0]):
        er2 = symmetric_tv_mode_product(
            A, res['res_x'][jj], m-1) - res['res_lbd'][jj] * res['res_x'][jj]
        if (np.sum(np.abs(er2)) > tol):
            print('err jj=%')
            
    all_eig, n_runs = find_all_unitary_eigenpair(
        all_eig=None, eig_cnt=None, A=A, max_itr=max_itr,
        max_test=int(1e6), tol=tol)
    print(np.sum(np.abs(all_eig.lbd)) - np.sum(np.abs(res['res_lbd'])))
    print(np.sum(np.abs(all_eig.x)) - np.sum(np.abs(res['res_x'])))
    print(np.sum(np.abs(all_eig.is_real)) - np.sum(np.abs(res['res_is_real'])))
    print(np.sum(np.abs(all_eig.is_self_conj)) -
          np.sum(np.abs(res['res_is_self_conj'])))


if __name__ == '__main__':
    np.random.seed(0)
    k = 8
    m = 3
    max_err = 1e-10
    max_itr = 200
    n_test = 1000

    summ = _test_eigen_tensor(k, m, max_err, max_itr, n_test)
    print(summ[[a for a in summ.columns if 'err' in a]].describe())
    print(summ[[a for a in summ.columns if 'iter' in a]].describe())
    print(summ[[a for a in summ.columns if 'lbd' in a]].describe())
    print(summ[[
        'o_ncm_time', 'schur_time', 'ray_time', 'schur_cheb_time']].describe())

    n = 2
    m = 3
    tol = 1e-10
    max_itr = 200
    A = utils.generate_symmetric_tensor(n, m)
    summ.to_csv('/tmp/c.csv')
