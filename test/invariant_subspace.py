from __future__ import print_function
import numpy as np
from scipy.linalg import solve, solve_sylvester
from lagrange_rayleigh.core.constraints import base_constraints
from lagrange_rayleigh.core.Lagrangian import Lagrangian
from lagrange_rayleigh.core.Lagrangian import state_keys as sk


def make_symmetric(ar):
    s = ar.shape[0]
    s0 = int((np.sqrt(8 * s + 1) - 1) / 2)
    h = np.zeros((s0, s0), dtype=float)
    cnt = 0
    for i in range(s0):
        h[i, :i+1] = ar[cnt:cnt + i + 1]
        h[:i, i] = ar[cnt:cnt + i]
        cnt += i+1
    return h


def flatten_symmetric(mat):
    s0 = mat.shape[0]
    ar = np.zeros(((s0*(s0+1)) // 2), dtype=float)
    cnt = 0
    for i in range(s0):
        ar[cnt:cnt+i+1] = mat[i, :i+1]
        cnt += i+1
    return ar


class invariant_subspace(Lagrangian):
    """Lagrangian for the invariant subspace
    problem for general matrix
    """
    is_explicit = True

    def __init__(self, A):
        self.A = A

    def F(self, x):
        return np.dot(self.A, x)

    def calc_J_F(self, x):
        return self.A

    def L_lambda(self, lbd):
        return -1 * np.dot(self._x, lbd)

    def L(self, x, lbd):
        return self.F(x) - np.dot(x, lbd)

    def eigen_solver(self, b):
        return solve_sylvester(self.A, -1 * self['LBD_J_H'], b)

    def hth_solver(self, b):
        return solve(self['HTH'], b)

    def tensor_solve(self, T, m):
        # m is a matrix of size (s0, s1)
        # T is a tensor of size (s0, s1, s0, s1)
        # thus T defines a map from array(s0, s1) to
        # array(s0, s1). we find the inverse of this
        vm = flatten_symmetric(m)
        return make_symmetric(np.linalg.solve(T, vm))
    
    def j_c_zeta_solver(self, zeta, rhs):
        T = self['J_C_zeta']
        if isinstance(rhs, list):
            return (self.tensor_solve(T, rhsa) for rhsa in rhs)
        else:
            return self.tensor_solve(T, rhs)

    def j_c_H_solver(self, rhs):
        # j_c_h = np.dot(self._x.T, self['H'], 1)
        T = self['J_C_zeta']
        return self.tensor_solve(T, rhs)
        
    def J_F(self, d_x):
        return np.dot(self.A, d_x)

    def J_H(self, d_x, d_lbd):
        return np.dot(d_x, d_lbd)

    def J_H2(self, d_x, d_lbd):
        return np.zeros((d_x.shape[0], d_lbd.shape[1]), dtype=float)

    def J_RAYLEIGH(self, d_x):
        p1 = np.dot(d_x.T, self._x)
        p2 = np.dot(p1 + p1.T, self['HFX'])
        
        p3 = np.dot(d_x.T, self['F'])
        return p3 + p3.T - p2

    def calc_H(self, x):
        return self._x
        
    def J_C(self, d_x):
        return self.constraints.J_C(d_x)
        
    def J_C2(self, d_x):
        return self.constraints.J_C2(d_x)

    def J_F2(self, d_x):
        return np.zeros_like(d_x)

    def calc_states(self, x, lbd, states=[sk.explicit_newton_raphson]):
        self._x = x
        self._state['H'] = self._x
        self._state['F'] = self.F(x)

        if 'RAYLEIGH' in states:
            self._state['HTH'] = np.dot(x.T, x)
            self._state['HFX'] = np.dot(x.T, self['F'])
            self._state['RAYLEIGH'] = self.hth_solver(self['HFX'])
            lbd = self._state['RAYLEIGH']

        self._lbd = lbd
        self.constraints.calc_states(x)
        for k in self.constraints.available_states:
            self._state[k] = self.constraints[k]

        self._state['LBD_J_H'] = lbd
        self._state['L'] = self['F'] - np.dot(x, lbd)

        s0 = lbd.shape[0]
        sa = (s0 * (s0 + 1)) / 2
        
        zeta = np.zeros((x.shape[0], x.shape[1], sa))
        for i in range(lbd.shape[0]):
            for j in range(i+1):
                hd = np.zeros_like(x)
                hd[:, i] = x[:, j]
                if j < i:
                    hd[:, j] = x[:, i]
                ff = self.eigen_solver(hd)
                zeta[:, :, (i * (i+1)) // 2 + j] = ff
                
        self._state['Lx_inv_H'] = zeta
        T = np.zeros((sa, sa))
        for i in range(s0):
            for j in range(i+1):
                ix = (i*(i+1)) // 2 + j
                T[:, ix] = flatten_symmetric(
                    self.J_C(zeta[:, :, ix]))
                    
        self._state['J_C_zeta'] = T

        self._state['PI'] = np.eye(x.shape[0]) - np.dot(x, x.T)
        
        """
        if 'J_C_H' in states:
            J_C_H = np.zeros((sa, sa))
            for i in range(s0):
                for j in range(i+1):
        """

    def tensordot(self, zeta, nu):
        return np.tensordot(zeta, flatten_symmetric(nu), 1)

    """
    def iterative_solver(self, x, lbd, w=1.):
        min_diff = 1e-4
        eta = 0
        mu = 0
        diff = 10.
        while np.linalg.norm(diff) > min_diff:
            eta = self.eigen_solver(self.L_lambda(mu) - self['F'])
            diff = self.J_C(eta)
            mu = mu - w * diff
        return eta, mu
    """

    
class stiefel(base_constraints):
    def __init__(self, n, p, k=1):
        super(stiefel, self).__init__(
            (n, p), (p, p))
        self._n = n
        self._p = p
        self._k = k

    def calc_states(self, x):
        self._x = x
        self._state['C'] = self.equality(x)
        # J_C2 is just 2 * the identity matrix
        # which we dont need to save a state
                                    
    def equality(self, x):
        return np.dot(x.T, x) - np.eye(self._p)

    def J_C(self, d_x):
        r = np.dot(self._x.T, d_x)
        return r + r.T

    def J_C2(self, d_x):
        return 2 * np.dot(d_x.T, d_x)

    def retraction(self, x, u):
        """ Retract to the Stiefel using the qr decomposition of x + g.
        """
        if self._k == 1:
            # Calculate 'thin' qr decomposition of x + u
            q, r = np.linalg.qr(x + u)
            # Unflip any flipped signs
            x_new = np.dot(q, np.diag(np.sign(np.sign(np.diag(r))+.5)))
        else:
            x_new = x + u
            for i in xrange(self._k):
                q, r = np.linalg.qr(x_new[i])
                x_new[i] = np.dot(q, np.diag(np.sign(np.sign(np.diag(r))+.5)))
        return x_new


def _test_invariance_subspace():
    from lagrange_rayleigh.core.utils import gen_random_symmetric
    from lagrange_rayleigh.core.solver import explicit_newton_raphson
    from lagrange_rayleigh.core.solver import rayleigh_quotient_iteration
    # from lagrange_rayleigh.core.solver import explicit_chebyshev, rayleigh_chebyshev

    np.random.seed(0)
    n = 7
    p = 2
    A = gen_random_symmetric(n)

    sc = stiefel(n, p)
    ei, v = np.linalg.eigh(A)
    
    x0a = np.random.randint(-5, 5, (n, p)) / 10.
    x0 = sc.retraction(x0a, 0)
    iv = invariant_subspace(A)
    iv.constraints = sc
    lbd0 = np.dot(x0.T, np.dot(A, x0))
    
    res_e = explicit_newton_raphson(
        iv, x0, lbd0, feasible=False, verbose=True)
    print(res_e)

    x0a = np.random.randint(-5, 5, (n, p)) / 10.
    x0 = sc.retraction(x0a, 0)
    lbd0 = np.dot(x0.T, np.dot(A, x0))
    res_f = explicit_newton_raphson(
        iv, x0, lbd0, max_err=1e-3,
        feasible=True, verbose=True)
    print(res_f)
    
    x0a = np.random.randint(-5, 5, (n, p)) / 10.
    x0 = sc.retraction(x0a, 0)
    lbd0 = np.dot(x0.T, np.dot(A, x0))
    res_ray = rayleigh_quotient_iteration(iv, x0, verbose=True)
    print(res_ray)
    print(iv.L(res_ray['x'], res_ray['lbd']))

    def make_y(x, lbd):
        y = np.zeros(n*p + (p * (p+1)) // 2)
        y[:n*p] = x.T.reshape(-1)
        y[n*p:] = flatten_symmetric(lbd)
        return y

    def LL(y):
        """
        y[:n*p] is x
        y[n:] is lbd
        """
        x = y[:n*p].reshape(p, n).T
        ret = np.zeros_like(y)
        L = np.dot(A, x) - np.dot(x, make_symmetric(y[n*p:]))
        ret[:n*p] = L.T.reshape(-1)
        ret[n*p:] = flatten_symmetric(np.dot(x.T, x) - np.eye(p))
        return ret

    def parse_y(y):
        x = y[:n*p].reshape(p, n).T
        lbd = make_symmetric(y[n*p:])
        return x, lbd

    def check_y(y):
        x, lbd = parse_y(y)
        return np.dot(A, x) - np.dot(x, lbd), np.dot(x.T, x) - np.eye(p)
                                            
    def J_LL(y):
        x, lbd = parse_y(y)
        jac = np.zeros((y.shape[0], y.shape[0]))
        for i in range(p):
            jac[i*n:(i+1)*n, i*n:(i+1)*n] = A
        for i1 in range(p):
            for i2 in range(p):
                for i_n in range(n):
                    jac[i1*n+i_n, i2*n+i_n] -= lbd[i1, i2]

        # derivatives with respect to lbd:

        for i1 in range(p):
            si1 = (i1*(i1+1)) // 2
            for i2 in range(i1):
                jac[i2*n:i2*n+n, n*p+si1+i2] = -x[:, i1]
                jac[i1*n:i1*n+n, n*p+si1+i2] = -x[:, i2]
            jac[i1*n:i1*n+n, n*p+si1+i1] = -x[:, i1]

        # derivatives of constraints
        for i1 in range(p):
            si = (i1 * (i1 + 1)) // 2
            jac[n*p+si+i1, i1*n:i1*n+n] = 2 * x[:, i1]
            for i2 in range(i1):
                jac[n*p + si + i2, i1*n:i1*n+n] = x[:, i2]
                jac[n*p + si + i2, i2*n:i2*n+n] = x[:, i1]

        return jac
        
    from scipy.optimize import fsolve
    y0 = np.zeros(n*p + (p * (p+1)) // 2)
    y0[:n*p] = x0.T.reshape(-1)
    y0[n*p:] = flatten_symmetric(lbd0)
    ee = 1e-4
    jl1 = np.zeros((y0.shape[0], y0.shape[0]))
    for ix in range(y0.shape[0]):
        # v = np.arange(1, y0.shape[0]+1) * 1e-4
        v = np.zeros_like(y0)
        v[ix] = ee
        jl1[:, ix] = (LL(y0 + v) - LL(y0)) / ee
    ijl1 = np.linalg.inv(jl1)
    res = fsolve(LL, y0)
    res1 = fsolve(LL, y0, fprime=J_LL)

    yy = y0.copy()
    i = 0
    l1 = np.array([100.])
    while np.linalg.norm(l1) > 1e-4:
        l1 = LL(yy)
        j1 = J_LL(yy)
        diff = np.linalg.solve(j1, l1)
        yy = yy - diff
        print('i=%d l1=%s' % (i, l1))
        i += 1
    xres, lbdres = parse_y(yy)
    
    print(res)
    print(res1)
    
if __name__ == '__main__':
    _test_invariance_subspace()
    
