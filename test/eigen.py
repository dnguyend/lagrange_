from __future__ import print_function
import numpy as np
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

    # now do rayleigh
    res_ray = rayleigh_quotient_iteration(e, x0, verbose=True)
    print(res_ray)
    print(e.L(res_ray['x'], res_ray['lbd']))

    # now do rayleigh chebyshev            
    res_ray_cheb = rayleigh_chebyshev(e, x0, verbose=True)
    print(res_ray_cheb)
    
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

    res_ray = rayleigh_quotient_iteration(e, x0, verbose=True)
    print(res_ray)
    print(e.L(res_ray['x'], res_ray['lbd']))

    # now do rayleigh chebyshev        
    res_ray_cheb = rayleigh_chebyshev(e, x0, verbose=True)
    print(res_ray_cheb)

    
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

    res_ray = rayleigh_quotient_iteration(e, x0, verbose=True)
    print(res_ray)
    print(e.L(res_ray['x'], res_ray['lbd']))

    # now do rayleigh chebyshev        
    res_ray_cheb = rayleigh_chebyshev(e, x0, verbose=True)
    print(e.L(res_ray_cheb['x'], res_ray_cheb['lbd']))
    print(res_ray_cheb)

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

    res_ray = rayleigh_quotient_iteration(en, x0, verbose=True)
    print(res_ray)
    print(en.L(res_ray['x'], res_ray['lbd']))

    # now do rayleigh chebyshev        
    res_ray_cheb = rayleigh_chebyshev(en, x0, verbose=True)
    print(en.L(res_ray_cheb['x'], res_ray_cheb['lbd']))
    print(res_ray_cheb)

    # now change method dont use HT
    def calc_J_RAYLEIGH(self, x):
        xsq = np.sum(x * x)
        return (- 2. / (xsq * xsq) * self['RAYLEIGH'] * x.T + np.dot(
            x.T / xsq, self._args['A'] + self._args['A'].T)).reshape(1, -1)
    eigen_ht_lagrange.calc_J_RAYLEIGH = calc_J_RAYLEIGH    

    en1 = eigen_ht_lagrange(B, z)
    en1.constraints = hyperplan
    res_ray1 = rayleigh_quotient_iteration(en1, x0, verbose=True)
    print(res_ray1)
    print(en1.L(res_ray1['x'], res_ray1['lbd']))
    
    res_ray_cheb1 = rayleigh_chebyshev(en1, x0, verbose=True)
    print(en1.L(res_ray_cheb1['x'], res_ray_cheb1['lbd']))
    print(res_ray_cheb1)
    

if __name__ == '__main__':
    _test_eigenvector()
    _test_eigen_linear()
    
    
        
        
