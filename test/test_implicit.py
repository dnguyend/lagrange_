from __future__ import print_function

import numpy as np
from scipy.optimize import newton

from lagrange_rayleigh.core.Lagrangian import state_keys as sk
from lagrange_rayleigh.core.Lagrangian import state_requests as sr
from lagrange_rayleigh.core.vector_lagrangian import implicit_vector_lagrangian
from lagrange_rayleigh.core.parametrized_constraints import parametrized_constraints
from lagrange_rayleigh.core.constraints import base_constraints
from lagrange_rayleigh.core.solver import implicit_newton_raphson
from lagrange_rayleigh.core.utils import gen_random_real_eigen
from lagrange_rayleigh.core.utils import gen_random_symmetric_pos


class vector_test_implicit(implicit_vector_lagrangian):
    """ run an explicit with implicit solver
    """
    def __init__(self, A, calc_H, calc_J_H):
        self._A = A
        self._shape_in = (A.shape[0], 0)
        self._shape_out = (A.shape[0], 0)
        self._H_ = calc_H
        self._J_H_ = calc_J_H
        
    def L(self, x, lbd):
        return np.dot(self._A, x) +\
            np.dot(self['L_lambda'], lbd)
    
    def calc_states(self, x, lbd, states=sr[sk.implicit_newton_raphson]):
        self._state['L_lambda'] = self.calc_L_lambda(x, lbd)
        self._state['J_H'] = self._J_H_(x)
        self._state['LBD_J_H'] = np.tensordot(
            self._state['J_H'].T, lbd, [1, 0])
        super(vector_test_implicit, self).calc_states(x, lbd)

    def calc_L_x(self, x, lbd):
        return self._A - self._state['LBD_J_H']
    
    def calc_L_lambda(self, x, lbd):
        return -self._H_(x)

    
class quadratic_eigen(implicit_vector_lagrangian):
    """ run quadratic eigen
    problem: P(lambda) x = 0
    constraint x.T x = 1
    """
    is_explicit = True
    
    def __init__(self, M, C, K):
        self.M = M
        self.C = C
        self.K = K
        self._shape_in = (M.shape[0], 0)
        self._shape_out = (M.shape[0], 0)

    def calc_states(self, x, lbd, states=sr[sk.implicit_newton_raphson]):
        self._state['P_matrix'] = self.M * lbd * lbd + self.C * lbd + self.K
        self._state['P_matrix_prime'] = self.M * (2 * lbd) + self.C
        super(quadratic_eigen, self).calc_states(x, lbd)

    def L(self, x, lbd):
        return np.dot(self._state['P_matrix'], x)

    def calc_L_x(self, x, lbd):
        return self['P_matrix']
    
    def L_x(self, d_x):
        return np.dot(self['L_x'], d_x)
    
    def calc_L_lambda(self, x, lbd):
        return np.dot(self['P_matrix_prime'], x).reshape(-1, 1)
    
    def L_lambda(self, d_lbd):
        return self['L_lambda'] * d_lbd
    

def _test_implcit():
    # allow for multiple constraints
    # F(X) is of simple form A X
    # more complex H

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
        
        def calc_H(x):
            ret = np.zeros((x.shape[0], 2))
            ret[:, 0] = (x * x)[:]
            ret[:, 1] = x[:]
            return ret

        def calc_J_H(x):
            ret = np.zeros((x.shape[0], 2, x.shape[0]))
            ret[:, 0, :] = np.diag(2 * x)
            ret[:, 1, :] = np.eye(x.shape[0])
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
        
        def calc_H(x):
            ret = np.zeros((x.shape[0], 2))
            ret[:-1, 0] = 1
            ret[:-2, 1] = np.arange(x.shape[0]-2)
            ret[-2, 0] = -1
            ret[-1, 1] = -1
            return ret

        def calc_J_H(x):
            ret = np.zeros((x.shape[0], 2, x.shape[0]))
            return ret

    n_var = 3
    funcs = [func1, func2]
    derivs = [deriv1, deriv2]

    n_func = len(funcs)
    n = n_func + n_var
    V = np.random.randint(-10, 10, (n, n))
    D = np.diag(np.random.randint(-10, 10, n))
    A = np.dot(np.dot(V, D), np.linalg.inv(V))

    x0a = np.random.randint(-5, 5, n_var) / 10.

    x0 = np.zeros((n_var+len(funcs)))
    x0[:n_var] = x0a
    x0[n_var:] = np.array([funcs[ix](x0a)
                           for ix in range(n_func)])
    lbd0 = np.array([1., 1.])
    e = vector_test_implicit(A, calc_H, calc_J_H)
    misc_constr = parametrized_constraints(
        (n, ), funcs, derivs)
    
    e.constraints = misc_constr
    
    res_e = implicit_newton_raphson(
        e, x0, lbd0, feasible=False, verbose=False)
    print(res_e)
    """
    ei, v = np.linalg.eig(A)
    print ei
    print v
    """
    # now do retraction:
    res_f = implicit_newton_raphson(
        e, res_e['x'] + .25,
        res_e['lbd']+.25, feasible=True, verbose=True)
    print(res_f)
    
    res_f = implicit_newton_raphson(
        e, x0, lbd0, feasible=True)
    print(res_f)
    

def _test_quadratic_eigen():
    k = 4
    M = gen_random_symmetric_pos(k)
    C = gen_random_symmetric_pos(k)
    K = gen_random_symmetric_pos(k)

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
    
    e = quadratic_eigen(M, C, K)
    e.constraints = sphere
    x0 = np.random.randn(k)
    x0 = x0 / np.linalg.norm(x0)
    lbd0 = np.array([1.])
    
    res_e = implicit_newton_raphson(
        e, x0, lbd0, feasible=False)
    print(res_e)

    # now do retraction:
    res_f = implicit_newton_raphson(
        e, x0,
        lbd0, feasible=True)

    print(res_f)

    # now do non symmetric
    M = gen_random_real_eigen(k)
    C = gen_random_real_eigen(k)
    K = gen_random_real_eigen(k)
    en = quadratic_eigen(M, C, K)
    
    en.constraints = sphere
    x0 = np.random.randn(k)
    x0 = x0 / np.linalg.norm(x0)
    lbd0 = np.array([1.])
    
    res_i_n = implicit_newton_raphson(
        en, x0, lbd0, feasible=False)
    print(res_i_n)

    # now do retraction:
    res_f_n = implicit_newton_raphson(
        en, x0,
        lbd0, feasible=True)

    print(res_f_n)
    
    
def test_rayleigh_implicit():
    # from scipy.sparse.linalg import lsmr
    np.random.seed(0)
    
    n = 3
    k = 1
    A = np.random.randn(n, n+k).astype(float)
    B = np.random.randn(n, n+k, n+k)
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    C = np.random.randn(n, n+k, n+k, n+k)
    C = C + np.transpose(C, (0, 1, 3, 2)) +\
        np.transpose(C, (0, 3, 2, 1)) +\
        np.transpose(C, (0, 2, 1, 3)) +\
        np.transpose(C, (0, 2, 3, 1)) +\
        np.transpose(C, (0, 3, 1, 2))
    C /= 6.
    
    x_r = np.zeros(n)
    x_r[0] = 1.
    lbd_r = 0.
    
    x0 = x_r + np.random.randn(n) * 1e-1
    x0 = x0 / np.sqrt(np.sum(x0*x0))

    def x_concat(x, lbd):
        return np.concatenate([x - x_r, np.array([lbd - lbd_r])])
    
    def L(x, lbd):
        # return B*x+np.sin(lbd*np.dot(A, x)+C*x)
        x_ = x_concat(x, lbd)
        return np.dot(A, x_) +\
            0.5 * np.tensordot(np.tensordot(B, x_, 1), x_, 1) +\
            np.tensordot(
                np.tensordot(np.tensordot(C, x_, 1), x_, 1), x_, 1) / 6.
        
    def Lx(x, lbd):
        x_ = x_concat(x, lbd)
        return A[:, :-k] + np.tensordot(B, x_, 1)[:, :-k] +\
            np.tensordot(np.tensordot(
                C, x_, 1), x_, 1)[:, :-k] / 2.
        """
        return np.diag(B) +\
            np.cos(lbd*np.dot(A, x)+C*x)[:, None]*(A*lbd+np.diag(C))
        """
        
    def Llbd(x, lbd):
        x_ = x_concat(x, lbd)
        return (A[:, -k:] + np.tensordot(B, x_, 1)[:, -k:] +
                np.tensordot(np.tensordot(
                    C, x_, 1), x_, 1)[:, -k:] / 2.).reshape(-1)
    
        """
        return np.dot(A, x)*np.cos(lbd*np.dot(A, x)+C*x)
        """
        
    def N(x, lbd):
        return np.sum(x * L(x, lbd))

    def Nx(x, lbd):
        return L(x, lbd) + np.dot(Lx(x, lbd).T, x)

    def Nlbd(x, lbd):
        return np.sum(x * Llbd(x, lbd))
        
    def solve_for_lbd(x, prev_lbd):
        def fl(lbd):
            return N(x, lbd)

        def flprime(lbd):
            return Nlbd(x, lbd)
        
        return newton(fl, x0=prev_lbd, fprime=flprime)
        
    def iteration():
        # x0 = np.random.randn(n)
        # x0 = x0 / np.linalg.norm(x0)
        x = x0.copy()
        lbd = .05
        for i in range(20):
            lbd = solve_for_lbd(x, lbd)
            LX = L(x, lbd)
            LxX = Lx(x, lbd)
            LlbdX = Llbd(x, lbd)
            l1 = np.sum(LlbdX*LlbdX)
            Px = np.eye(n) -\
                np.dot(LlbdX.reshape(-1, 1), 1./l1 * LlbdX.reshape(1, -1))
            PLx = np.zeros((n+1, n), dtype=float)
            PLx[-1, :] = x
            PLx[:-1, :] = np.dot(Px, LxX)
            PL = np.dot(Px, LX)
            PL = np.concatenate([PL, np.array([0])])
            eta = -np.linalg.solve(
                np.dot(PLx.T, PLx), np.dot(PLx.T, PL))
            x = x + eta
            x = x / np.linalg.norm(x)
            print(eta)
            print(L(x, lbd))

    def test_derive(x, lbd):
        LX = L(x, lbd)
        hh = x * x * 1e-4
        h = 1.e-4
        print(L(x,  lbd + h) - LX)
        print(Llbd(x, lbd))

        print(L(x + hh,  lbd) - LX)
        print(np.dot(Lx(x, lbd), hh))
        
        x_ = x_concat(x, lbd)
        print(_bb(x_))
        hh_ = x_ * x_ * 1e-4
        print(_bb(x_ + hh_) - _bb(x_))
        print(np.dot(_bb_dev(x_), hh_))

        print(_cc(x_ + hh_) - _cc(x_))
        print(np.dot(_cc_dev(x_), hh_))
        
    def _bb(x_):
        return 0.5 * np.tensordot(np.tensordot(B, x_, 1), x_, 1)

    def _bb_dev(x_):
        return np.tensordot(B, x_, 1)

    def _cc(x_):
        return np.tensordot(np.tensordot(
            np.tensordot(C, x_, 1), x_, 1), x_, 1) / 6.

    def _cc_dev(x_):
        return np.tensordot(
            np.tensordot(C, x_, 1), x_, 1) / 2.


def _test_rayleigh_implicit_multi_constraints():
    from scipy.optimize import fsolve
    np.random.seed(0)

    k = 2
    nn = np.random.randint(2, 5, (k))
    n = np.sum(nn)
    cnn = np.concatenate([np.zeros((1)), np.cumsum(nn)]).astype(int)
    
    def retract(x):
        for i in range(k):
            x[cnn[i]:cnn[i+1]] = x[cnn[i]:cnn[i+1]] /\
                np.linalg.norm(x[cnn[i]:cnn[i+1]])
        return x

    A = np.random.randn(n, n+k).astype(float)
    B = np.random.randn(n, n+k, n+k)
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    C = np.random.randn(n, n+k, n+k, n+k)
    C = C + np.transpose(C, (0, 1, 3, 2)) +\
        np.transpose(C, (0, 3, 2, 1)) +\
        np.transpose(C, (0, 2, 1, 3)) +\
        np.transpose(C, (0, 2, 3, 1)) +\
        np.transpose(C, (0, 3, 1, 2))
    C /= 6.

    # constraints are product of spheres
    x_r = np.zeros(n)
    x_r[0] = 1.
    x_r[np.cumsum(nn)[:-1]] = 1
    lbd_r = np.zeros((k))
    
    x0 = x_r + np.random.randn(n) * 1e-1
    x0 = retract(x0)

    def x_concat(x, lbd):
        return np.concatenate([x - x_r, lbd - lbd_r])
    
    def L(x, lbd):
        x_ = x_concat(x, lbd)
        return np.dot(A, x_) +\
            0.5 * np.tensordot(np.tensordot(B, x_, 1), x_, 1) +\
            np.tensordot(
                np.tensordot(np.tensordot(C, x_, 1), x_, 1), x_, 1) / 6.
        
    def Lx(x, lbd):
        x_ = x_concat(x, lbd)
        return A[:, :-k] + np.tensordot(B, x_, 1)[:, :-k] +\
            np.tensordot(np.tensordot(
                C, x_, 1), x_, 1)[:, :-k] / 2.
        
    def Llbd(x, lbd):
        x_ = x_concat(x, lbd)
        return A[:, -k:] + np.tensordot(B, x_, 1)[:, -k:] +\
            np.tensordot(np.tensordot(
                C, x_, 1), x_, 1)[:, -k:] / 2.
        
    def N(x, lbd):
        xL = x * L(x, lbd)
        return np.array([np.sum(xL[cnn[i]:cnn[i+1]]) for i in range(k)])

    def Nx(x, lbd):
        ret = np.zeros((k, n), dtype=float)
        l1 = Lx(x, lbd)
        for i in range(k):
            ret[i, cnn[i]:cnn[i+1]] = L(x, lbd)[cnn[i]:cnn[i+1]]
            ret[i, :] += np.sum(x[cnn[i]:cnn[i+1]][:, None] *
                                l1[cnn[i]:cnn[i+1], :], axis=0)
        return ret

    def Nlbd(x, lbd):
        xLlbd = x[:, None] * Llbd(x, lbd)
        ret = np.zeros((k, k))
        for i in range(k):
            ret[i, :] = np.sum(xLlbd[cnn[i]:cnn[i+1], :], axis=0)
        return ret
        
    def solve_for_lbd(x, prev_lbd):
        def fl(lbd):
            return N(x, lbd)

        def flprime(lbd):
            return Nlbd(x, lbd)
        
        return fsolve(fl, x0=prev_lbd, fprime=flprime)
        
    def iteration():
        # x0 = np.random.randn(n)
        # x0 = x0 / np.linalg.norm(x0)
        x = x0.copy()
        lbd = np.full((k), .05)
        for i in range(20):
            lbd = solve_for_lbd(x, lbd)
            LX = L(x, lbd)
            LxX = Lx(x, lbd)
            LlbdX = Llbd(x, lbd)
            l1 = np.dot(LlbdX.T, LlbdX)
            Px = np.eye(n) -\
                np.dot(LlbdX, np.linalg.solve(l1, LlbdX.T))
            PLx = np.zeros((n+k, n), dtype=float)
            PLx[-k, :] = x
            PLx[:-k, :] = np.dot(Px, LxX)
            PL = np.dot(Px, LX)
            PL = np.concatenate([PL, np.zeros((k))])
            eta = -np.linalg.solve(
                np.dot(PLx.T, PLx), np.dot(PLx.T, PL))
            x = retract(x + eta)
            print(eta)
            print(L(x, lbd))

    def test_derive(x, lbd):
        LX = L(x, lbd)
        hh = x * x * 1e-4
        h = np.random.randint(2, 3, k) * 1.e-4
        print(L(x,  lbd + h) - LX)
        print(np.dot(Llbd(x, lbd), h))

        print(L(x + hh,  lbd) - LX)
        print(np.dot(Lx(x, lbd), hh))

        NX = N(x, lbd)
        print(N(x,  lbd + h) - NX)
        print(np.dot(Nlbd(x, lbd), h))
        print(N(x + hh,  lbd) - NX)
        print(np.dot(Nx(x, lbd), hh))
        
    
if __name__ == '__main__':
    _test_implcit()
    _test_quadratic_eigen()
    _test_rayleigh_implicit_multi_constraints()
        
