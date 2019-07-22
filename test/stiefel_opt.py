from __future__ import print_function
import numpy as np
# from scipy.linalg import solve, solve_sylvester
from scipy.sparse.linalg import lsmr
from lagrange_rayleigh.core.constraints import base_constraints
import lagrange_rayleigh.core.utils as utils


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


def rayleigh_iteration_lsmr(F, J_F, s, x0, niter=10):
    """ Solve the equation
    Pn L_x eta = - Pn F
    J_C eta = 0
    """
    x = x0.copy()
    n, p = x.shape
    TembT = utils.vech_embedding_tensor(p).T
    Ttrans = utils.transpose_tensor((n, p))
    err = 100
    i = 0
    while i < niter and (err > 1e-3):
        FX = F(x)
        Pn = np.eye(n * p) - 0.5 * (
            np.kron(np.dot(x, x.T), np.eye(p)) + np.dot(
                np.kron(x, x.T), Ttrans))
        # Pn = np.eye(n * p) - 0.5 * Pn 
        mPFX = -np.dot(Pn, FX.reshape(-1))
        JFX = utils.tensor_to_matrix_reshape(J_F(x), n, p)
        rayleigh = np.dot(x.T, FX)
        rayleigh = 0.5 * (rayleigh + rayleigh.T)
        PnLx = np.dot(Pn, JFX - np.kron(np.eye(n), rayleigh))

        J_C = np.dot(
            TembT,
            0.5 * (np.kron(x.T, np.eye(p)) + np.dot(
                np.kron(np.eye(p), x.T),
                Ttrans)))
        big_mat = np.concatenate([PnLx, J_C], axis=0)
        eta_ = lsmr(big_mat,
                    np.concatenate(
                        [mPFX.reshape(-1),
                         np.zeros(J_C.shape[0])]), maxiter=300)
        if eta_[1] > 2:
            print(eta_)
        x = s.retraction(x, eta_[0].reshape(x.shape))
        err_ = F(x) - np.dot(x, rayleigh)
        err = np.linalg.norm(err_)
        i += 1
    return x, F(x), rayleigh, eta_


def rayleigh_iteration(F, J_F, s, x0, niter=10):
    """ Solve the equation
    Pn L_x eta = - Pn F
    J_C eta = 0
    """
    x = x0.copy()
    n, p = x.shape
    # TembT = utils.vech_embedding_tensor(p).T
    # Ttrans = utils.transpose_tensor((n, p))
    err = 100
    ii = 0
    while ii < niter and (err > 1e-3):
        FX = F(x)
        xxT = np.dot(x, x.T)
        FXTx = np.dot(FX.T, x)
        mPFX = -FX + 0.5 * (np.dot(xxT, FX) + np.dot(x, FXTx))
        JFX = J_F(x)
        rayleigh = 0.5 * (FXTx + FXTx.T)
        u, _, _ = np.linalg.svd(x)
        x_perp = u[:, p:]
        # tangent space ix xA + x_perp B
        r_a, c_a = np.triu_indices(p, 1)
        r_b, c_b = np.indices((n-p, p))
        r_b = r_b.reshape(-1)
        c_b = c_b.reshape(-1)
        # dim_a = (p * (p - 1)) // 2
        # dim_b = n * (n - p)
        perp_base = np.zeros((n, p, len(r_a) + len(r_b)))
        perp_im = np.zeros_like(perp_base)
        for i in range(len(r_a)):
            perp_base[:, c_a[i], i] = x[:, r_a[i]]
            perp_base[:, r_a[i], i] = -x[:, c_a[i]]
            l1 = np.tensordot(JFX, perp_base[:, :, i])
            l1 = l1 - np.dot(perp_base[:, :, i], rayleigh)
            
            perp_im[:, :, i] = l1 -\
                0.5 * (np.dot(xxT, l1) + np.dot(x, np.dot(l1.T, x)))
        for i in range(len(r_b)):
            perp_base[:, c_b[i], len(r_a)+i] = x_perp[:, r_b[i]]
            l1 = np.tensordot(JFX, perp_base[:, :, len(r_a)+i])
            l1 = l1 - np.dot(perp_base[:, :, len(r_a)+i], rayleigh)
            
            perp_im[:, :, len(r_a) + i] = l1 -\
                0.5 * (np.dot(xxT, l1) + np.dot(x, np.dot(l1.T, x)))

        eta_ = lsmr(perp_im.reshape(n*p, -1),
                    mPFX.reshape(-1), maxiter=300)
        
        if eta_[1] > 2:
            print(eta_)
        eta = np.tensordot(perp_base, eta_[0], 1)
        x = s.retraction(x, eta)
        err_ = F(x) - np.dot(x, rayleigh)
        err = np.linalg.norm(err_)
        ii += 1
    return x, F(x), rayleigh, eta


def rayleigh_iteration_schur(F, J_F, s, x0, niter=10):
    """ Solve the equation
    Pn L_x eta = - Pn F
    J_C eta = 0
    """
    x = x0.copy()
    n, p = x.shape
    TembT = utils.vech_embedding_tensor(p).T
    Ttrans = utils.transpose_tensor((n, p))
    err = 100
    ii = 0
    while ii < niter and (err > 1e-3):
        FX = F(x)
        FXTx = np.dot(FX.T, x)

        rayleigh = 0.5 * (FXTx + FXTx.T)
        n_constraints = (p*(p+1)) // 2
        # stack FX and H to one big RHS matrix
        h_tensor = np.zeros((n, p, n_constraints), dtype=float)

        r_r, c_r = np.triu_indices(p)
        """
        for i in range(n_constraints):
            rhs[c_r[i], i+1] = x[:, r_r[i]]
            rhs[r_r[i], i+1] = x[:, c_r[i]]
        """
        h_tensor[:, c_r, np.arange(n_constraints)] = x[:, r_r]
        h_tensor[:, r_r, np.arange(n_constraints)] = x[:, c_r]
        rhs = np.concatenate([FX.reshape(-1, 1), h_tensor.reshape(n*p, -1)],
                             axis=1)
        JFX = utils.tensor_to_matrix_reshape(J_F(x), n, p)
        
        res_ = np.linalg.solve(JFX - np.kron(np.eye(n), rayleigh), rhs)
        J_C = np.dot(
            TembT,
            0.5 * (np.kron(x.T, np.eye(p)) + np.dot(
                np.kron(np.eye(p), x.T),
                Ttrans)))

        j_c_zeta = np.dot(J_C, res_[:, 1:])
        j_c_nu = np.dot(J_C, res_[:, 0])
        lbd_ = np.linalg.solve(j_c_zeta, j_c_nu)
        eta_ = np.dot(res_[:, 1:], lbd_) - res_[:, 0]
        x = s.retraction(x, eta_.reshape(n, p))
        err_ = F(x) - np.dot(x, rayleigh)
        err = np.linalg.norm(err_)
        ii += 1
    return x, F(x), rayleigh, eta_


def _test_stiefel_rayleigh():
    """
    """
    n = 10
    p = 4
    s = stiefel(n, p)
    x0a = np.random.randint(-10, 10, (n, p))
    x0 = s.retraction(x0a, 0)

    A = utils.gen_random_symmetric_pos(n*p)
    b = (np.random.randn(n*p)*5 + 1).reshape(n, p)

    def f(x):
        return .5 * np.dot(x.reshape(-1), np.dot(A, x.reshape(-1))) +\
            np.dot(b.reshape(-1), x.reshape(-1))

    def F(x):
        return (np.dot(A, x.reshape(-1)) + b.reshape(-1)).reshape(n, p)

    def J_F(x):
        """ return the (n, p, n, p) tensor representing differential of x"""
        """
        ret = np.zeros((n, p, n, p), dtype=float)
        for i in range(n):
            for j in range(p):
                ret[:, :, i, j] = A[:, i*p+j].reshape(n, p)
        return ret
        """
        return A.reshape(n, p, n, p)
    
    x = x0.copy()
    check_tensor = False
    if check_tensor:
        J1 = 0.5 * (np.kron(x.T, np.eye(p)) + np.dot(
            np.kron(np.eye(p), x.T), utils.transpose_tensor((n, p))))
        """
        J2a = 0.5 * (np.dot(np.kron(x, np.eye(p)),
        np.eye(p*p) + utils.transpose_tensor((p, p))))
        """
        J2 = np.kron(x, np.eye(p))
        # utils.vech_project_tensor(p)
        Temb = utils.vech_embedding_tensor(p)
        pJ1 = np.dot(Temb.T, J1)
        # pJ2a = np.dot(J2a, Temb)
        pJ2 = np.dot(J2, Temb)
        print(pJ1.T - 2 * pJ2)
        print(np.dot(pJ1, pJ2))
    x, FX, rayleigh, eta_ = rayleigh_iteration(F, J_F, s, x0, 1000)
    # print(f(x))
    # print(f(x0))
    print("test rayleigh iteration")
    for i in range(100):
        x1a = np.random.randn(n, p)
        x1 = s.retraction(x1a, 0)
        x1b, FXb, rr1, eta_ = rayleigh_iteration(F, J_F, s, x1, niter=1000)
        # print(FXb - F(x1b))
        # rr = np.dot(x1b.T, FXb)
        f1 = f(x1b)
        v1 = FXb - np.dot(x1b, rr1)
        e1 = np.linalg.norm(v1)
        if e1 > 1e-2:
            print("not converge err=%f" % e1)

        if False or (f1 < 80):
            print('%x and rayleigh')
            print(x1b, rr1)
            print('Residual values')
            print(v1)
            print('f value %f' % f(x1b))
    best_val = -1e6
    print("test rayleigh iteration with large number of different x0's")
    print("this just print a list of critical points")
    for i in range(10000):
        x2a = np.random.randn(n, p)
        x2 = s.retraction(x2a, 0)
        fx2 = f(x2)
        if fx2 > best_val:
            print('initial value=%s' % str(x2))
            print('fx %f' % fx2)
            x1c, FXc, rr2, eta_ = rayleigh_iteration(F, J_F, s, x2, niter=1000)
            fx3 = f(x1c)
            print('critial x x1c=%s' % str(x1c))
            print('value at critial x %f' % fx3)
            v2 = FXc - np.dot(x1c, rr2)
            e2 = np.linalg.norm(v2)
            print('Residual values %s' % str(v2))
            if e2 > 10:
                print("not converge err=%f" % e2)

            if fx3 > best_val:
                best_val = fx3
            if fx2 > best_val:
                best_val = fx2
            
    x3, FX3, rayleigh3, eta_3 = rayleigh_iteration_schur(F, J_F, s, x0, 1000)
    print("test rayleigh schur iteration")
    # print(f(x3))
    # print(f(x0))
    for i in range(100):
        x3a = np.random.randn(n, p)
        x3 = s.retraction(x3a, 0)
        x3c, FXc, rr3, eta_3 = rayleigh_iteration_schur(
            F, J_F, s, x3, niter=1000)
        f3 = f(x3c)
        v3 = FXc - np.dot(x3c, rr3)
        e2 = np.linalg.norm(v3)
        if e2 > 1e-2:
            print("not converge err=%f" % e2)

        if False or (f3 > 9000):
            print('%x and rayleigh')
            print(x3c, rr3)
            
            print('Residual values')
            print(v3)
            print('f value %f' % f(x1b))
            print(f(x3c))

    best_val = -1e6
    print("test rayleigh schur iteration with large number of different x0")
    for i in range(10000):
        x2_ = np.random.randn(n, p)
        x2 = s.retraction(x2_, 0)
        fx2 = f(x2)
        if fx2 > best_val:
            print('x2=%s' % str(x2))
            print(fx2)
            x3c, FXc, rr3, eta_3 = rayleigh_iteration_schur(
                F, J_F, s, x2, niter=1000)
            fx3 = f(x3c)
            print('x values x3c=%s' % str(x3c))
            print('fvalue % ' % fx3)
            v3 = FXc - np.dot(x3c, rr3)
            e3 = np.linalg.norm(v3)
            print('residual vector' % str(v3))
            if e3 > 10:
                print("not converge err=%f" % e3)

            if fx3 > best_val:
                best_val = fx3
            if fx2 > best_val:
                best_val = fx2
            

if __name__ == '__main__':
    _test_stiefel_rayleigh()
               
