from __future__ import print_function
import numpy as np
import pandas as pd
import time


import lagrange_rayleigh.core.utils as utils
from lagrange_rayleigh.core.vector_lagrangian import explicit_vector_lagrangian
from lagrange_rayleigh.core.constraints import base_constraints

from lagrange_rayleigh.core.solver import rayleigh_quotient_iteration
from lagrange_rayleigh.core.solver import rayleigh_chebyshev
from lagrange_rayleigh.core.eigen_tensor_solver import\
    orthogonal_newton_correction_method, schur_form_rayleigh,\
    symmetric_tv_mode_product, schur_form_rayleigh_chebyshev_linear,\
    schur_form_rayleigh_chebyshev


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
    ray_cheb_cnt = np.zeros(n_test, dtype=int)

    o_ncm_err = np.zeros(n_test)
    schur_err = np.zeros(n_test)
    ray_err = np.zeros(n_test)
    ray_cheb_err = np.zeros(n_test)

    o_ncm_lbd = np.zeros(n_test)
    schur_lbd = np.zeros(n_test)
    ray_lbd = np.zeros(n_test)
    ray_cheb_lbd = np.zeros(n_test)

    o_ncm_time = np.zeros(n_test)
    schur_time = np.zeros(n_test)
    ray_time = np.zeros(n_test)
    ray_cheb_time = np.zeros(n_test)
    
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
            s_x, s_lbd, ctr, converge = schur_form_rayleigh_chebyshev(
                A, max_itr, max_err, x_init=x0, do_chebyshev=True)

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
        res_ray_cheb = rayleigh_chebyshev(
            e, x0, max_err=max_err, max_iter=max_itr,
            verbose=False, exit_by_diff=True)
        t_end = time.time()
        ray_cheb_time[jj] = t_end - t_start

        ray_cheb_cnt[jj] = res_ray_cheb['n_iter']
        ray_cheb_lbd[jj] = res_ray_cheb['lbd']
        ray_cheb_err[jj] = np.linalg.norm(res_ray_cheb['err'])
        # print("doing raychev")
        # print(res_ray_cheb)
        # print(e.L(res_ray_cheb['x'], res_ray_cheb['lbd']))

    summ = pd.DataFrame(
        {
            'o_ncm_iter': o_ncm_cnt,
            'schur_iter': schur_cnt,
            'ray_iter': ray_cnt, 'ray_cheb_iter': ray_cheb_cnt,
            'o_ncm_err': o_ncm_err,
            'schur_err': schur_err,
            'ray_err': ray_err, 'ray_cheb_err': ray_cheb_err,
            'o_ncm_lbd': o_ncm_lbd,
            'schur_lbd': schur_lbd,
            'ray_lbd': ray_lbd,
            'ray_cheb_lbd': ray_cheb_lbd,
            'o_ncm_time': o_ncm_time,
            'schur_time': schur_time,
            'ray_time': ray_time,
            'ray_cheb_time': ray_cheb_time
        },
        columns=['o_ncm_iter', 'o_ncm_lbd', 'o_ncm_err', 'o_ncm_time',
                 'schur_iter', 'schur_lbd', 'schur_err', 'schur_time',
                 'ray_iter', 'ray_lbd', 'ray_err', 'ray_time',
                 'ray_cheb_iter', 'ray_cheb_lbd', 'ray_cheb_err',
                 'ray_cheb_time'])
    return summ

    
if __name__ == '__main__':
    np.random.seed(0)
    k = 20
    m = 3
    max_err = 1e-10
    max_itr = 200
    n_test = 100

    summ = _test_eigen_tensor(k, m, max_err, max_itr, n_test)
    print(summ[[a for a in summ.columns if 'err' in a]].describe())
    print(summ[[a for a in summ.columns if 'iter' in a]].describe())
    print(summ[[a for a in summ.columns if 'lbd' in a]].describe())
    print(summ[[
        'o_ncm_time', 'schur_time', 'ray_time', 'ray_cheb_time']].describe())
    summ.to_csv('/tmp/c.csv')
        
        
