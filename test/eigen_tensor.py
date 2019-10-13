from __future__ import print_function
import numpy as np

import lagrange_rayleigh.core.utils as utils
from lagrange_rayleigh.core.vector_lagrangian import explicit_vector_lagrangian
from lagrange_rayleigh.core.constraints import base_constraints

from lagrange_rayleigh.core.solver import rayleigh_quotient_iteration
from lagrange_rayleigh.core.solver import rayleigh_chebyshev


class eigen_tensor_lagrange(explicit_vector_lagrangian):
    def calc_H(self, x):
        return x[:, None]

    def F(self, x):
        v = self._args['A']
        for i in range(self._m-3):
            v = np.tensordot(v, x, axes=1)
        self._F2 = v / float(self._m)
        self._F1 = np.tensordot(self._F2, x, axes=1)
        self._F0 = np.tensordot(self._F1, x, axes=1)
        return self._F0
    
    def __init__(self, A):
        self._args = {'A': A}
        self._k = A.shape[0]
        self._m = len(A.shape)
        self._shape_in = (A.shape[0], 0)
        self._shape_out = (A.shape[0], 0)

    def calc_J_F(self, x):
        return self._F1 / float(self._m)

    def calc_J_H(self, x):
        return np.eye(x.shape[0]).reshape(
            x.shape[0], 1, x.shape[0])

    def J_F2(self, d_x):
        return np.dot(self._F1, d_x)

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

    
def _test_eigen_tensor():
    np.random.seed(0)
    k = 20
    m = 3
    max_err = 1e-10

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

    n_test = 500
    ray_cnt = np.zeros(n_test, dtype=int)
    ray_cheb_cnt = np.zeros(n_test, dtype=int)
    ray_err = np.zeros(n_test)
    ray_cheb_err = np.zeros(n_test)
    
    for jj in range(n_test):
        x0 = np.random.randn(k)
        x0 = x0 / np.linalg.norm(x0)

        # now do rayleigh    
        res_ray = rayleigh_quotient_iteration(
            e, x0, max_err=max_err, verbose=False,
            exit_by_diff=True)
        ray_cnt[jj] = res_ray['n_iter']
        ray_err[jj] = np.linalg.norm(res_ray['err'])
        # print("doing rayleigh")
        # print(res_ray)
        # print(e.L(res_ray['x'], res_ray['lbd']))

        # now do rayleigh chebyshev            
        res_ray_cheb = rayleigh_chebyshev(
            e, x0, max_err=max_err, verbose=False, exit_by_diff=True)
        ray_cheb_cnt[jj] = res_ray_cheb['n_iter']
        ray_cheb_err[jj] = np.linalg.norm(res_ray_cheb['err'])
        # print("doing raychev")
        # print(res_ray_cheb)
        # print(e.L(res_ray_cheb['x'], res_ray_cheb['lbd']))
    import pandas as pd
    summ = pd.DataFrame({'ray_iter': ray_cnt, 'ray_cheb_iter': ray_cheb_cnt,
                         'ray_err': ray_err, 'ray_cheb_err': ray_cheb_err},
                        columns=['ray_iter', 'ray_err',
                                 'ray_cheb_iter', 'ray_cheb_err'])
    print(summ.describe())

    
if __name__ == '__main__':
    _test_eigen_tensor()
    
    
        
        
