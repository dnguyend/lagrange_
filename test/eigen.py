from __future__ import print_function
import numpy as np
import time
from scipy.linalg import null_space
from numpy.linalg import norm, solve

import lagrange_rayleigh.core.utils as utils
from lagrange_rayleigh.core.vector_lagrangian import explicit_vector_lagrangian
from lagrange_rayleigh.core.constraints import base_constraints
from lagrange_rayleigh.core.solver import explicit_newton_raphson
from lagrange_rayleigh.core.solver import rayleigh_quotient_iteration
from lagrange_rayleigh.core.solver import rayleigh_chebyshev
from lagrange_rayleigh.core.solver import explicit_chebyshev


class eigen_vector_lagrange(explicit_vector_lagrangian):
    def calc_H(self, x):
        return x[:, None]

    def F(self, x):
        return np.dot(self._args['A'], x)
    
    def __init__(self, A):
        self._args = {'A': A}
        self._shape_in = (A.shape[0], 0)
        self._shape_out = (A.shape[0], 0)

    def calc_J_F(self, x):
        return self._args['A']

    def calc_J_H(self, x):
        return np.eye(x.shape[0]).reshape(
            x.shape[0], 1, x.shape[0])

    def J_F2(self, d_x):
        return np.zeros(d_x.shape)

    def J_C(self, d_x):
        return np.dot(
            self._state['J_C'], d_x)

    def calc_J_F2(self, x):
        return np.zeros_like(x)

    def calc_J_H2(self, x):
        pass

    def J_H2(self, d_x, d_lbd):
        return np.zeros((d_x.shape[0]))

    def J_C2(self, d_x):
        return 2 * np.dot(d_x.T, d_x).reshape(1)
    
    def calc_J_RAYLEIGH(self, x):
        return (- 2 * self['RAYLEIGH'] * x.T + np.dot(
            x.T, self._args['A'] + self._args['A'].T)).reshape(1, -1)


class eigen_ht_lagrange(explicit_vector_lagrangian):
    """ Eigen problem so H = x. However
    HT could be choosen differently
    """
    def calc_H(self, x):
        return x[:, None]

    def F(self, x):
        return np.dot(self._args['A'], x)
    
    def __init__(self, A, z, HT=None):
        self._args = {'A': A, 'z': z}
        self._shape_in = (A.shape[0], 0)
        self._shape_out = (A.shape[0], 0)
        if HT is not None:
            self.calc_HT = HT
            self.has_HT = True

    def calc_J_F(self, x):
        return self._args['A']

    def calc_J_H(self, x):
        return np.eye(x.shape[0]).reshape(
            x.shape[0], 1, x.shape[0])

    def J_F2(self, d_x):
        return np.zeros(d_x.shape)

    def J_C(self, d_x):
        return np.dot(
            self._state['J_C'], d_x)

    def calc_J_F2(self, x):
        return np.zeros_like(x)

    def J_H2(self, d_x, d_lbd):
        return np.zeros((d_x.shape[0]))

    def calc_J_H2(self, x):
        pass

    def J_C2(self, d_x):
        return np.zeros((1), dtype=float)

    def calc_J_RAYLEIGH(self, x):
        """This assumes HT with Rayleigh
        (z.T x)^{-1} z^TAx
        """
        return (np.dot(self._args['z'].T, self._args['A']) -
                self['RAYLEIGH'] * self._args['z'].T).reshape(1, -1)

    
def _test_eigenvector():
    np.random.seed(0)
    k = 5

    x0 = np.random.randint(-5, 5, k) / 10.
    x0 = x0 / np.linalg.norm(x0)

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

    A = utils.gen_random_symmetric(k)
    e = eigen_vector_lagrange(A)
    e.constraints = sphere
    res_e = explicit_newton_raphson(
        e, x0, np.array([1.]), feasible=False, verbose=True)
    print(res_e)
    ei, v = np.linalg.eigh(A)
    print(ei)
    print(v)
    # now do retraction:
    for i in range(10):
        x1 = np.random.randint(-5, 5, k) / 10.
        x1 = x1 / np.linalg.norm(x1)

        res_f = explicit_newton_raphson(
            e, x1, np.array([1.]), feasible=True, verbose=False)
        print(res_f)
        print(e.L(res_f['x'], res_f['lbd']))

    # now do explicit_chebyshev
    
    res_chev = explicit_chebyshev(
        e, x0, np.array([1.]), feasible=False, verbose=True)
    print(res_chev)

    res_chev_f = explicit_chebyshev(
        e, x0, np.array([1.]), feasible=True, verbose=True)
    print(res_chev_f)

    for i in range(10):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)

        # now do rayleigh
        res_ray = rayleigh_quotient_iteration(e, x0, verbose=False)
        print(res_ray)
        # print(e.L(res_ray['x'], res_ray['lbd']))
        # now do rayleigh chebyshev
        res_ray_cheb = rayleigh_chebyshev(e, x0, verbose=False)
        print(res_ray_cheb)

    n_test = 1000
    tbl = np.zeros((n_test, 6))
    for i in range(n_test):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)

        # now do rayleigh
        res_ray = rayleigh_quotient_iteration(e, x0, verbose=False)
        tbl[i, 0] = res_ray['n_iter']
        tbl[i, 1] = res_ray['lbd']
        tbl[i, 2] = np.linalg.norm(res_ray['err'])
        # print(res_ray)
        # print(e.L(res_ray['x'], res_ray['lbd']))
        # now do rayleigh chebyshev
        res_ray_cheb = rayleigh_chebyshev(e, x0, verbose=False)
        # print(res_ray_cheb)
        tbl[i, 3] = res_ray_cheb['n_iter']
        tbl[i, 4] = res_ray_cheb['lbd']
        tbl[i, 5] = np.linalg.norm(res_ray_cheb['err'])
    print(np.mean(tbl, axis=0))
            
    # now do the non symmetric case
    B = utils.gen_random_real_eigen(k)
    e = eigen_vector_lagrange(B)
    e.constraints = sphere
    res_e = explicit_newton_raphson(
        e, x0, np.array([1.]), feasible=False, verbose=True)
    print(res_e)
    ei, v = np.linalg.eig(B)
    print(ei)
    print(v)
    # now do retraction:
    res_f = explicit_newton_raphson(
        e, x0, np.array([1.]), feasible=True, verbose=True)
    print(res_f)

    res_chev = explicit_chebyshev(
        e, x0, np.array([1.]), feasible=False, verbose=True)
    print(res_chev)

    res_chev = explicit_chebyshev(
        e, x0, np.array([1.]), feasible=True, verbose=True)
    print(res_chev)

    n_test = 10
    for i in range(n_test):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)

        print("Doing Rayleigh")
        res_ray = rayleigh_quotient_iteration(e, x0, verbose=False)
        print(res_ray)
        # print(e.L(res_ray['x'], res_ray['lbd']))

        # now do rayleigh chebyshev
        print("Doing Rayleigh Chebysev")
        res_ray_cheb = rayleigh_chebyshev(e, x0, verbose=False, cutoff=1e-1)
        print(res_ray_cheb)

    n_test = 1000
    tbl2 = np.full((n_test, 6), np.nan)
    from lagrange_rayleigh.core import solver
    solver._MAX_ERR = 1e-10
    solver._MAX_ITER = 200
    for i in range(n_test):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)
        B = utils.gen_random_real_eigen(k)
        e = eigen_vector_lagrange(B)
        e.constraints = sphere

        res_ray = rayleigh_quotient_iteration(
            e, x0, max_err=1e-8, verbose=False)
        if np.linalg.norm(res_ray['err']) < .01:
            tbl2[i, 0] = res_ray['n_iter']
            tbl2[i, 1] = res_ray['lbd']
            tbl2[i, 2] = np.linalg.norm(res_ray['err'])

        # print("Doing Rayleigh Chebysev")
        try:
            res_ray_cheb = rayleigh_chebyshev(
                e, x0, max_err=1e-8, verbose=False, cutoff=2e-1)
            if np.linalg.norm(res_ray_cheb['err']) < .01:
                tbl2[i, 3] = res_ray_cheb['n_iter']
                tbl2[i, 4] = res_ray_cheb['lbd']
                tbl2[i, 5] = np.linalg.norm(res_ray_cheb['err'])
        except Exception:
            pass
    print(np.nanmean(tbl2, axis=0))

    
def _test_eigen_linear():
    """
    with linear constraint
    we try two ways to do Rayleigh / Rayleigh Chebysev
    with HT = H.T and HT = z.T
    """
    np.random.seed(0)
    k = 5

    x0 = np.random.randint(-5, 5, k) / 10.
    x0 = x0 / np.linalg.norm(x0)

    # need ||z|| = 1
    z = x0.copy()

    def hyperplan_func(x):
        return np.dot(z.T, x) - 1

    def hyperplan_jacobian(x):
        return z.reshape(1, -1)

    def hyperplan_retraction(x, u):
        # return x + u
        x0 = x + u
        return x0 + (1 - np.dot(z, x0)) * z
    
    hyperplan = base_constraints(
        shape_in=(k,),
        shape_constraint=(1,),
        equality=hyperplan_func)

    hyperplan.set_analytics(
        J_C=hyperplan_jacobian,
        retraction=hyperplan_retraction)

    A = utils.gen_random_symmetric(k)
    
    def HT(x):
        return z.T
    
    e = eigen_ht_lagrange(A, z, HT)
    e.constraints = hyperplan
    res_e = explicit_newton_raphson(
        e, x0, np.array([1.]), feasible=False)
    print(res_e)
    ei, v = np.linalg.eigh(A)
    print(ei)
    print(v)
    # now do retraction:
    res_f = explicit_newton_raphson(
        e, x0, np.array([1.]), feasible=True)
    print(res_f)
    
    res_cheb = explicit_chebyshev(
        e, x0, np.array([1.]), feasible=False, verbose=True)
    print(res_cheb)

    res_cheb = explicit_chebyshev(
        e, x0, np.array([1.]), feasible=True, verbose=True)
    print(e.L(res_cheb['x'], res_cheb['lbd']))
    print(res_cheb)

    for i in range(10):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)

        res_ray = rayleigh_quotient_iteration(e, x0, verbose=False)
        print(res_ray)
        # print(e.L(res_ray['x'], res_ray['lbd']))

        # now do rayleigh chebyshev        
        res_ray_cheb = rayleigh_chebyshev(e, x0, verbose=False, cutoff=3e-1)
        # print(e.L(res_ray_cheb['x'], res_ray_cheb['lbd']))
        print(res_ray_cheb)

    n_test = 1000
    tbl = np.full((n_test, 6), np.nan)

    for i in range(n_test):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)
        try:
            res_ray = rayleigh_quotient_iteration(e, x0, verbose=False)
            tbl[i, 0] = res_ray['n_iter']
            tbl[i, 1] = res_ray['lbd']
            tbl[i, 2] = np.linalg.norm(res_ray['err'])
        except Exception:
            pass

        # now do rayleigh chebyshev
        try:
            res_ray_cheb = rayleigh_chebyshev(e, x0, verbose=False, cutoff=3e-1)
            tbl[i, 3] = res_ray_cheb['n_iter']
            tbl[i, 4] = res_ray_cheb['lbd']
            tbl[i, 5] = np.linalg.norm(res_ray_cheb['err'])
        except Exception:
            pass
    print(np.nanmean(tbl, axis=0))

    # example 2 non symmetric
    B = utils.gen_random_real_eigen(k)
    en = eigen_ht_lagrange(B, z, HT)
    en.constraints = hyperplan
    res_e = explicit_newton_raphson(
        en, x0, np.array([1.]), feasible=False)
    print(res_e)
    print(en.L(res_e['x'], res_e['lbd']))
    print(en.constraints.equality(res_e['x']))

    ei, v = np.linalg.eig(B)
    print(ei)
    print(v)
    # now do retraction:
    res_f = explicit_newton_raphson(
        en, x0, np.array([1.]), feasible=True, verbose=True)
    print(res_f)
    print(en.L(res_f['x'], res_f['lbd']))
    print(en.constraints.equality(res_f['x']))

    res_chev = explicit_chebyshev(
        en, x0, np.array([1.]), feasible=False, verbose=True)
    print(res_chev)
    print(en.L(res_chev['x'], res_chev['lbd']))
    print(en.constraints.equality(res_chev['x']))

    res_chev = explicit_chebyshev(
        en, x0, np.array([1.]), feasible=True, verbose=True)
    print(res_chev)
    print(en.L(res_chev['x'], res_chev['lbd']))
    print(en.constraints.equality(res_chev['x']))

    for i in range(10):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)
        print("Do rayleigh")
        res_ray = rayleigh_quotient_iteration(en, x0, verbose=False)
        print(res_ray)
        # print(en.L(res_ray['x'], res_ray['lbd']))

        # now do rayleigh chebyshev
        print("Do rayleigh chebyshev")
        res_ray_cheb = rayleigh_chebyshev(en, x0, verbose=False, cutoff=1.e-1)
        print(en.L(res_ray_cheb['x'], res_ray_cheb['lbd']))
        print(res_ray_cheb)

    n_test = 1000
    tbl2 = np.full((n_test, 6), np.nan)

    for i in range(n_test):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)
        try:
            res_ray = rayleigh_quotient_iteration(en, x0, verbose=False)
            tbl2[i, 0] = res_ray['n_iter']
            tbl2[i, 1] = res_ray['lbd']
            tbl2[i, 2] = np.linalg.norm(res_ray['err'])
        except Exception:
            pass
        try:
            res_ray_cheb = rayleigh_chebyshev(en, x0, verbose=False, cutoff=2.e-1)
            tbl2[i, 3] = res_ray_cheb['n_iter']
            tbl2[i, 4] = res_ray_cheb['lbd']
            tbl2[i, 5] = np.linalg.norm(res_ray_cheb['err'])
        except Exception:
            pass

    print(np.nanmean(tbl2, axis=0))

    # now change method dont use HT
    def calc_J_RAYLEIGH(self, x):
        xsq = np.sum(x * x)
        return (- 2. / (xsq * xsq) * self['RAYLEIGH'] * x.T + np.dot(
            x.T / xsq, self._args['A'] + self._args['A'].T)).reshape(1, -1)
    eigen_ht_lagrange.calc_J_RAYLEIGH = calc_J_RAYLEIGH    

    en1 = eigen_ht_lagrange(B, z)
    en1.constraints = hyperplan

    for i in range(10):
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)

        res_ray1 = rayleigh_quotient_iteration(en1, x0, verbose=False)
        print(res_ray1)
        # print(en1.L(res_ray1['x'], res_ray1['lbd']))

        res_ray_cheb1 = rayleigh_chebyshev(
            en1, x0, verbose=False, cutoff=5e-1)
        print(en1.L(res_ray_cheb1['x'], res_ray_cheb1['lbd']))
        print(res_ray_cheb1)


def linear_rqi_xTx(x0, A, z, do_cheb=False,
                   max_iter=200, max_err=1e-8, cutoff=3e-1):
    """in this version use (x.Tx) for left inverse
    """
    x = x0.copy()
    k = A.shape[0]
    for i in range(max_iter):
        xTx = x.T @ x
        Ax = A @ x
        xTAx = x.T @ Ax
        lbd = xTAx / xTx
        zeta = solve(A - lbd * np.eye(k), x)
        err = norm(A @ x - lbd * x)
        # if err < max_err or (norm(zeta) * max_err > 10):
        if err < max_err:
            if err < max_err:
                return x, lbd, i, err, True
            else:
                return x, lbd, i, err, False

        nu = x + lbd * zeta
        lbd_s = (z.T @ nu) / (z.T @ zeta)
        eta = -nu + zeta * lbd_s

        if do_cheb and (norm(eta) < cutoff):
            J_R = -2*lbd/xTx*x@eta+x.T@(A.T+A)@eta/(xTx)
            T = solve(A - lbd * np.eye(k), eta * J_R)
            tau = T + eta - zeta / (z @ zeta) * (z @ (T+eta))
            x_n = (x + tau) / (z.T @ (x + tau))
        else:
            x_n = zeta / (z.T @ zeta)
        R = norm(x_n - x)
        x = x_n
        if R < max_err:
            break

    return x, lbd, i, err, err < max_err


def linear_rqi_zTx(x0, A, z, do_cheb=False,
                   max_iter=200, max_err=1e-8, cutoff=3e-1):
    """in this version use (z.Tx) for left inverse
    """
    x = x0.copy()
    k = A.shape[0]
    tau = None
    R = 1
    for i in range(max_iter):
        zTx = z.T @ x
        Ax = A @ x
        zTAx = z.T @ Ax
        lbd = zTAx / zTx
        zeta = solve(A - lbd * np.eye(k), x)
        err = norm(A @ x - lbd * x)
        # if err < max_err or (norm(zeta) * max_err > 10):
        if err < max_err:
            if err < max_err:
                return x, lbd, i, err, True
            else:
                return x, lbd, i, err, False
        nu = x + lbd * zeta
        lbd_s = (z.T @ nu) / (z.T @ zeta)
        eta = -nu + zeta * lbd_s

        if do_cheb and (norm(eta) < cutoff):
            J_R = -lbd*z.T@eta+z.T@A@eta
            T1 = solve(A - lbd * np.eye(k), eta * J_R)
            tau = T1 + eta - zeta / (z @ zeta) * (z @ (T1+eta))
            x_n = (x + tau) / (z.T @ (x+tau))
        else:
            x_n = zeta / (z.T @ zeta)
        R = norm(x-x_n)
        """
        print('i=%d tau=%s x=%s R=%f lbd=%f' % (
            i, str(tau), str(x), R, lbd))
        """
        x = x_n

        if R < max_err:
            break

        # if err < max_err:
        #    break
    return x, lbd, i, err, err < max_err


def linear_rqi_zTx_tangent(x0, A, z, do_cheb=False,
                           max_iter=200, max_err=1e-8, cutoff=3e-1):
    """in this version use (z.Tx) for left inverse
    z.T x = 1 is used for the constraint
    """
    # from scipy.linalg import null_space
    # Y = null_space(z)
    k = A.shape[0]
    Z = np.eye(k)
    # the Chebysev step:
    z = Z[:, 0]
    Y = Z[:, 1:]
    v = np.zeros((k-1))
    # y = Y u
    x_v = x0 / (z @ x0)
    success = False
    zTAY = z.T @ A @ Y
    tau = v
    err = 1
    for i in range(max_iter):
        x_v = x_v + Y @ tau
        p1 = np.eye(k) - x_v.reshape(-1, 1) @ z.reshape(1, -1)
        Y_p1 = Y.T @ p1
        lbd = z.T @ A @ x_v
        Rmd = (A @ x_v - lbd * x_v)
        err = norm(Rmd)
        if err <= max_err:
            success = True
            break
        pj_L = Y_p1 @ Rmd
        pj_Lx = Y_p1 @ (A - lbd * np.eye(k)) @ Y
        v = solve(pj_Lx, - pj_L)
        if do_cheb and (norm(v) < cutoff):
            J_R = zTAY @ v
            tau = v + solve(pj_Lx, Y_p1 @ Y @ v * J_R)
        else:
            tau = v
        
    if not success:
        x_v = x_v + Y @ v
        # p1 = np.eye(k) - x_v.reshape(-1, 1) @ z.reshape(1, -1)
        # Y_p1 = Y.T @ p1
        lbd = z.T @ A @ x_v
        Rmd = (A @ x_v - lbd * x_v)
        err = norm(Rmd)
        if err <= max_err:
            success = True
    # x, lbd, n_iter, err, converge
    return x_v, lbd, i, err, success


def linear_rqi_xTx_tangent(x0, A, z, do_cheb=False,
                           max_iter=200, max_err=1e-8, cutoff=3e-1):
    """(x.T x)^{-1} x.T is used for the left inverse
    z.T x = 1 is used for the constraint
    pH = I_k - (x x.T) / (x.T @ x)
    pH @ x_v = 0
    so a basis for pH is (null x_v.T).T
    """
    k = A.shape[0]
    Z = np.eye(k)
    # the Chebysev step:
    z = Z[:, 0]
    Y = Z[:, 1:]
    v = np.zeros((k-1))
    # y = Y u
    symAY = (A.T+A) @ Y
    x_v = x0 / (z @ x0)
    success = False
    tau = v
    err = 1
    for i in range(max_iter):
        x_v = x_v + Y @ tau
        xTx = x_v @ x_v
        pH = np.eye(k) - x_v.reshape(-1, 1) @ x_v.reshape(1, -1) / xTx
        YH = null_space(x_v.reshape(1, -1))
        Y_pH = YH.T @ pH
        lbd = (x_v.T @ A @ x_v) / xTx
        Rmd = (A @ x_v - lbd * x_v)
        err = norm(Rmd)
        if err <= max_err:
            success = True
            break
        pj_L = Y_pH @ Rmd
        pj_Lx = Y_pH @ (A - lbd * np.eye(k)) @ Y
        v = solve(pj_Lx, - pj_L)
        if do_cheb and (norm(v) < cutoff):
            J_R = -2*lbd/xTx*x_v@(Y@v)+x_v.T@symAY@v/(xTx)
            tau = v + solve(pj_Lx, Y_pH @ Y @ v * J_R)
        else:
            tau = v
        
    if not success:
        x_v = x_v + Y @ v
        # p1 = np.eye(k) - x_v.reshape(-1, 1) @ z.reshape(1, -1)
        # Y_p1 = Y.T @ p1
        xTx = x_v @ x_v
        lbd = x_v.T @ A @ x_v / xTx
        Rmd = (A @ x_v - lbd * x_v)
        err = norm(Rmd)
        if err <= max_err:
            success = True
    # x, lbd, n_iter, err, converge
    return x_v, lbd, i, err, success
        

def test_linear():
    k = 50
    n_test = 500

    tbl2 = np.full((n_test, 32), np.nan)
    z = np.array([1] + (k-1)*[0])
    max_iter = 200
    max_err = 1e-10
    for i in range(n_test):
        A = utils.gen_random_real_eigen(k)
        x0 = np.random.randint(-5, 5, k) / 10.
        x0 = x0 / np.linalg.norm(x0)
        try:
            t_start = time.time()
            x, lbd, n_iter, err, converge = linear_rqi_xTx_tangent(
                x0, A, z, do_cheb=False, max_iter=max_iter,
                max_err=max_err, cutoff=3e-1)
            t_end = time.time()
            tbl2[i, 0] = n_iter
            tbl2[i, 1] = lbd
            tbl2[i, 2] = err
            tbl2[i, 3] = t_end - t_start
        except Exception:
            pass
        try:
            t_start = time.time()
            x, lbd, n_iter, err, converge = linear_rqi_xTx_tangent(
                x0, A, z, do_cheb=True, max_iter=max_iter,
                max_err=max_err, cutoff=10.e-1)
            t_end = time.time()
            tbl2[i, 4] = n_iter
            tbl2[i, 5] = lbd
            tbl2[i, 6] = err
            tbl2[i, 7] = t_end - t_start

        except Exception:
            pass
        try:
            t_start = time.time()
            x, lbd, n_iter, err, converge = linear_rqi_xTx(
                x0, A, z, do_cheb=False, max_iter=max_iter,
                max_err=max_err, cutoff=3e-1)
            t_end = time.time()
            tbl2[i, 8] = n_iter
            tbl2[i, 9] = lbd
            tbl2[i, 10] = err
            tbl2[i, 11] = t_end - t_start
        except Exception:
            pass
        try:
            t_start = time.time()
            x, lbd, n_iter, err, converge = linear_rqi_xTx(
                x0, A, z, do_cheb=True, max_iter=max_iter,
                max_err=max_err, cutoff=10.e-1)
            t_end = time.time()
            tbl2[i, 12] = n_iter
            tbl2[i, 13] = lbd
            tbl2[i, 14] = err
            tbl2[i, 15] = t_end - t_start
        except Exception:
            pass

        # zTx
        try:
            t_start = time.time()
            x, lbd, n_iter, err, converge = linear_rqi_zTx_tangent(
                x0, A, z, do_cheb=False, max_iter=max_iter,
                max_err=max_err, cutoff=3e-1)
            t_end = time.time()
            tbl2[i, 16] = n_iter
            tbl2[i, 17] = lbd
            tbl2[i, 18] = err
            tbl2[i, 19] = t_end - t_start
        except Exception:
            pass
        try:
            t_start = time.time()
            x, lbd, n_iter, err, converge = linear_rqi_zTx_tangent(
                x0, A, z, do_cheb=True, max_iter=max_iter,
                max_err=max_err, cutoff=5.e-1)
            t_end = time.time()
            tbl2[i, 20] = n_iter
            tbl2[i, 21] = lbd
            tbl2[i, 22] = err
            tbl2[i, 23] = t_end - t_start

        except Exception:
            pass
        try:
            t_start = time.time()
            x, lbd, n_iter, err, converge = linear_rqi_zTx(
                x0, A, z, do_cheb=False, max_iter=max_iter,
                max_err=max_err, cutoff=3e-1)
            t_end = time.time()
            tbl2[i, 24] = n_iter
            tbl2[i, 25] = lbd
            tbl2[i, 26] = err
            tbl2[i, 27] = t_end - t_start
        except Exception:
            pass
        try:
            t_start = time.time()
            x, lbd, n_iter, err, converge = linear_rqi_zTx(
                x0, A, z, do_cheb=True, max_iter=max_iter,
                max_err=max_err, cutoff=10.e-1)
            t_end = time.time()
            tbl2[i, 28] = n_iter
            tbl2[i, 29] = lbd
            tbl2[i, 30] = err
            tbl2[i, 31] = t_end - t_start
        except Exception:
            pass
        
    tbl3 = tbl2.copy()
    tbl3[np.where(~(tbl2[:, 2] < .05))[0], :4] = np.nan
    tbl3[np.where(~(tbl2[:, 6] < .05))[0], 4:8] = np.nan
    tbl3[np.where(~(tbl2[:, 10] < .05))[0], 8:12] = np.nan
    tbl3[np.where(~(tbl2[:, 14] < .05))[0], 12:16] = np.nan

    tbl3[np.where(~(tbl2[:, 18] < .05))[0], 16:20] = np.nan
    tbl3[np.where(~(tbl2[:, 22] < .05))[0], 20:24] = np.nan
    tbl3[np.where(~(tbl2[:, 26] < .05))[0], 24:28] = np.nan
    tbl3[np.where(~(tbl2[:, 30] < .05))[0], 28:32] = np.nan
    print(np.nanmean(tbl3, axis=0))
    import pandas as pd
    column0 = ['%s_iter %s_lbd %s_err %s_time' % (
        a, a, a, a) for a in [
            'xtx_tgt', 'xtx_chev_tgt', 'xtx_amb', 'xtx_chev_amb',
            'ztx_tgt', 'ztx_chev_tgt', 'ztx_amb', 'ztx_chev_amb',
        ]]
    columns = np.squeeze([a.split() for a in column0]).reshape(-1)
    pDF = pd.DataFrame(tbl3, columns=columns)
    for tag in ['iter', 'lbd', 'err', 'time']:
        print(pDF[[a for a in pDF.columns if a.endswith(tag)]].describe().T)


if __name__ == '__main__':
    _test_eigenvector()
    _test_eigen_linear()
    
    
        

