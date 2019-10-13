from __future__ import print_function

import numpy as np
from lagrange_rayleigh.core.vector_lagrangian import explicit_vector_lagrangian
from lagrange_rayleigh.core.parametrized_constraints import parametrized_constraints

from lagrange_rayleigh.core.solver import explicit_newton_raphson
from lagrange_rayleigh.core.solver import rayleigh_quotient_iteration
from lagrange_rayleigh.core.solver import explicit_chebyshev
from lagrange_rayleigh.core.solver import rayleigh_chebyshev


class vector_test_lagrange(explicit_vector_lagrangian):
    def calc_H(self, x):
        """initiated after creation
        """

    def calc_J_H(self, x):
        """initiated after creation
        """

    def calc_J_H2(self, x):
        """initiated after creation
        """
        
    def calc_J_F2(self, x):
        """initiated after creation
        """
        return None
            
    def F(self, x):
        return np.dot(self._args['A'], x)
    
    def __init__(self, A, H, calc_J_H, calc_J_H2=None):
        self._args = {'A': A}
        self._shape_in = (A.shape[0], 0)
        self._shape_out = (A.shape[0], 0)
        self.calc_H = H
        self.calc_J_H = calc_J_H
        if calc_J_H2 is not None:
            self.calc_J_H2 = calc_J_H2
            
    def calc_J_F(self, x):
        return self._args['A']

    def J_F2(self, d_x):
        return np.zeros(d_x.shape)

    def J_C(self, d_x):
        return np.dot(
            self._state['J_C'], d_x)

    def calc_J_RAYLEIGH(self, x):
        pass

    def J_RAYLEIGH(self, d_x):
        jhx = self.J_H(d_x, self['RAYLEIGH'])
        p2 = np.dot(self['H'].T, jhx)        
        p3 = p2.T + p2
        p4 = np.dot(
            self.J_H(d_x, np.linalg.inv(self['HTH'])).T, self['F'])
        ret = self.hth_solver(np.dot(self['H'].T, self.J_F(d_x)))
        return ret + p4 - self.hth_solver(p3)
    
    """
    def calc_J_RAYLEIGH(self, x):
        ret = np.zeros((x.shape[0], 2, x.shape[0], x.shape[0]))
        return ret

    def calc_J_RAYLEIGH(x):
        return np.dot(
            self['HTH'], np.dot(self['H'].T, A))
    """


class vector_nonlinear_lagrange(explicit_vector_lagrangian):
    def calc_H(self, x):
        """initiated after creation
        """

    def calc_J_H(self, x):
        """initiated after creation
        """

    def calc_J_H2(self, x):
        """initiated after creation
        """
        
    def calc_J_F2(self, x):
        """initiated after creation

        ret = np.zeros(
            (x.shape[0], x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            ret[:, :, i] = - np.dot(
                self._args['B'],
                np.dot(self._args['B'], np.sin(
                    np.dot(self._args['B'], x))))
        """
        return None
            
    def F(self, x):
        return np.dot(self._args['A'], x) + np.sin(
            np.dot(self._args['B'], x))
    
    def __init__(self, A, B, H, calc_J_H, calc_J_H2=None):
        self._args = {'A': A, 'B': B}
        self._shape_in = (A.shape[0], 0)
        self._shape_out = (A.shape[0], 0)
        self.calc_H = H
        self.calc_J_H = calc_J_H
        if calc_J_H2 is not None:
            self.calc_J_H2 = calc_J_H2
            
    def calc_J_F(self, x):
        return self._args['A'] + np.diag(
            np.dot(
                self._args['B'], np.cos(
                    np.dot(self._args['B'], x))))

    def J_F2(self, d_x):
        B = self._args['B']
        return - np.dot(B, np.dot(B, np.sin(
            np.dot(B, self._x)))) * d_x * d_x

    def J_C(self, d_x):
        return np.dot(
            self._state['J_C'], d_x)

    def calc_J_RAYLEIGH(self, x):
        pass

    def J_RAYLEIGH(self, d_x):
        jhx = self.J_H(d_x, self['RAYLEIGH'])
        p2 = np.dot(self['H'].T, jhx)        
        p3 = p2.T + p2
        p4 = np.dot(
            self.J_H(d_x, np.linalg.inv(self['HTH'])).T, self['F'])
        ret = self.hth_solver(np.dot(self['H'].T, self.J_F(d_x)))
        return ret + p4 - self.hth_solver(p3)
    
    """
    def calc_J_RAYLEIGH(self, x):
        ret = np.zeros((x.shape[0], 2, x.shape[0], x.shape[0]))
        return ret

    """


def _test_parametrized_constraints():
    """ Allow for multiple constraints
    F(X) is of simple form A X
    more complex H
    May not converge if see bad results try a different initial point
    """

    nonlinear_constraint = True
    if nonlinear_constraint:
        def func1(x):
            return np.sum(-2 * x + np.sin(x))

        def func2(x):
            return np.sum(x + np.cos(x))

        def deriv1(x):
            return -2 + np.cos(x)

        def deriv2(x):
            return 1 - np.sin(x)

        def second_deriv1(x):
            return np.diag(-np.sin(x))

        def second_deriv2(x):
            return np.diag(-np.cos(x))

        def H(x):
            ret = np.zeros((x.shape[0], 2))
            ret[:, 0] = (x * x)[:]
            ret[:, 1] = x[:]
            return ret

        def calc_J_H(x):
            ret = np.zeros((x.shape[0], 2, x.shape[0]))
            ret[:, 0, :] = np.diag(2 * x)
            ret[:, 1, :] = np.eye(x.shape[0])
            return ret

        def calc_J_H2(x):
            ret = np.zeros((x.shape[0], 2, x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                ret[i, 0, i, i] = 2
            return ret
            
    else:
        def func1(x):
            return np.sum(x) - 1

        def func2(x):
            return np.sum(x * np.arange(x.shape[0])) - 1

        def deriv1(x):
            return x

        def deriv2(x):
            return np.arange(x.shape[0])

        def second_deriv1(x):
            return np.eye(x.shape[0])

        def second_deriv2(x):
            return np.zeros((x.shape[0], x.shape[0]), dtype=float)

        def H(x):
            ret = np.zeros((x.shape[0], 2))
            ret[:-1, 0] = 1
            ret[:-2, 1] = np.arange(x.shape[0]-2)
            ret[-2, 0] = -1
            ret[-1, 1] = -1
            return ret

        def calc_J_H(x):
            ret = np.zeros((x.shape[0], 2, x.shape[0]))
            return ret

        def calc_J_H2(x):
            return np.zeros((x.shape[0], 2, x.shape[0], x.shape[0]))
        
    n_var = 3
    funcs = [func1, func2]
    derivs = [deriv1, deriv2]

    n_func = len(funcs)
    n = n_func + n_var
    V = np.random.randint(-10, 10, (n, n))
    D = np.diag(np.random.randint(-10, 10, n))
    A = np.dot(np.dot(V, D), np.linalg.inv(V))

    def gen_start_point():
        x1a = np.random.randint(-5, 5, n_var) / 10.

        x1 = np.zeros((n_var+len(funcs)))
        x1[:n_var] = x1a
        x1[n_var:] = np.array([funcs[ix](x1a)
                               for ix in range(n_func)])
        return x1
        
    x0 = gen_start_point()    
    lbd0 = np.array([1., 1.])
    e = vector_test_lagrange(A, H, calc_J_H)
    misc_constr = parametrized_constraints(
        (n, ), funcs, derivs)
    
    e.constraints = misc_constr
    
    res_e = explicit_newton_raphson(
        e, x0, lbd0, verbose=True, feasible=False)
    print(res_e)
    """
    ei, v = np.linalg.eig(A)
    print ei
    print v
    """
    # now do retraction:
    x1 = gen_start_point()
    res_f = explicit_newton_raphson(
        e, x1,
        res_e['lbd']+.4, max_err=1e-3, feasible=True, verbose=True)

    print(res_f)

    # res_ray = rayleigh_quotient_iteration(
    # e, x0)
    for i in range(10):
        x1 = gen_start_point()
        res_ray = rayleigh_quotient_iteration(
            e, x1)
        print('x1= %s ' % str(x1))
        print(res_ray)

    # now try chebyshev
    second_derivs = [second_deriv1, second_deriv2]
    echeb = vector_test_lagrange(A, H, calc_J_H, calc_J_H2)
    cheb_constr = parametrized_constraints(
        (n, ), funcs, derivs, second_derivs)
    echeb.constraints = cheb_constr

    x0 = res_ray['x'] + .3
    lbd0 = res_ray['lbd'] + .3
    res_cheb = explicit_chebyshev(
        echeb, x0, lbd0, feasible=False)
    print(res_cheb)

    # rayleigh_chebyshev
    for i in range(10):
        x1 = gen_start_point()
        res_ray_chev = rayleigh_chebyshev(echeb, x1)
        print(res_ray_chev)
    

def _test_nonlinear_parametrized_constraints():
    """ Allow for multiple constraints
    F(X) is of simple form A X + sin(Bx)
    more complex H
    May not converge if see bad results try a different initial point
    """

    nonlinear_constraint = True
    if nonlinear_constraint:
        def func1(x):
            return np.sum(-2 * x + np.sin(x))

        def func2(x):
            return np.sum(x + np.cos(x))

        def deriv1(x):
            return -2 + np.cos(x)

        def deriv2(x):
            return 1 - np.sin(x)

        def second_deriv1(x):
            return np.diag(-np.sin(x))

        def second_deriv2(x):
            return np.diag(-np.cos(x))

        def H(x):
            ret = np.zeros((x.shape[0], 2))
            ret[:, 0] = (x * x)[:]
            ret[:, 1] = x[:]
            return ret

        def calc_J_H(x):
            ret = np.zeros((x.shape[0], 2, x.shape[0]))
            ret[:, 0, :] = np.diag(2 * x)
            ret[:, 1, :] = np.eye(x.shape[0])
            return ret

        def calc_J_H2(x):
            ret = np.zeros((x.shape[0], 2, x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                ret[i, 0, i, i] = 2
            return ret
            
    else:
        def func1(x):
            return np.sum(x) - 1

        def func2(x):
            return np.sum(x * np.arange(x.shape[0])) - 1

        def deriv1(x):
            return x

        def deriv2(x):
            return np.arange(x.shape[0])

        def second_deriv1(x):
            return np.eye(x.shape[0])

        def second_deriv2(x):
            return np.zeros((x.shape[0], x.shape[0]), dtype=float)

        def H(x):
            ret = np.zeros((x.shape[0], 2))
            ret[:-1, 0] = 1
            ret[:-2, 1] = np.arange(x.shape[0]-2)
            ret[-2, 0] = -1
            ret[-1, 1] = -1
            return ret

        def calc_J_H(x):
            ret = np.zeros((x.shape[0], 2, x.shape[0]))
            return ret

        def calc_J_H2(x):
            return np.zeros((x.shape[0], 2, x.shape[0], x.shape[0]))
        
    n_var = 3
    funcs = [func1, func2]
    derivs = [deriv1, deriv2]

    n_func = len(funcs)
    n = n_func + n_var
    V = np.random.randint(-10, 10, (n, n))
    D = np.diag(np.random.randint(-10, 10, n))
    A = np.dot(np.dot(V, D), np.linalg.inv(V))
    B = np.random.randint(-10, 10, (n, n)).astype(float) * 1e-4

    def gen_start_point():
        x1a = np.random.randint(-5, 5, n_var) / 10.

        x1 = np.zeros((n_var+len(funcs)))
        x1[:n_var] = x1a
        x1[n_var:] = np.array([funcs[ix](x1a)
                               for ix in range(n_func)])
        return x1
        
    x0 = gen_start_point()    
    lbd0 = np.array([1., 1.])
    e = vector_nonlinear_lagrange(A, B, H, calc_J_H)
    misc_constr = parametrized_constraints(
        (n, ), funcs, derivs)
    
    e.constraints = misc_constr
    
    x0 = gen_start_point()
    res_e = explicit_newton_raphson(
        e, x0, lbd0, feasible=False)
    print(res_e)
    """
    ei, v = np.linalg.eig(A)
    print ei
    print v
    """
    # now do retraction:
    x1 = gen_start_point()
    res_f = explicit_newton_raphson(
        e, x1,
        res_e['lbd']+.3, max_err=1e-3, feasible=True, verbose=True)

    print(res_f)

    # res_ray = rayleigh_quotient_iteration(
    # e, x0)
    for i in range(10):
        x1 = gen_start_point()
        res_ray = rayleigh_quotient_iteration(
            e, x1)
        print('x1= %s ' % str(x1))
        print(res_ray)

    # now try chebyshev
    second_derivs = [second_deriv1, second_deriv2]
    echeb = vector_nonlinear_lagrange(A, B, H, calc_J_H, calc_J_H2)
    cheb_constr = parametrized_constraints(
        (n, ), funcs, derivs, second_derivs)
    echeb.constraints = cheb_constr

    x0 = res_ray['x'] + .3
    lbd0 = res_ray['lbd'] + .3
    res_cheb = explicit_chebyshev(
        echeb, x0, lbd0, feasible=False)
    print(res_cheb)
    
    # rayleigh_chebyshev
    for i in range(10):
        x1 = gen_start_point()
        res_ray_chev = rayleigh_chebyshev(echeb, x1)
        print(res_ray_chev)

        
if __name__ == '__main__':
    _test_parametrized_constraints()
    _test_nonlinear_parametrized_constraints()
