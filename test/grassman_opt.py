import numpy as np


class grassmann(object):
    def __init__(self, n, p, k=1):
        self._n = n
        self._p = p
        self._k = k
        self._state = {}

    def retraction(self, x, u):
        """ Retract to the Grass using the qr decomposition of x + g.
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


def gen_tridiag(n, c, d, e):
    ret = np.eye(n)
    # idx = np.concatenate([np.arange(n), np.arange(n)].reshape(2, -1)
    ret[(np.arange(n), np.arange(n))] = d
    ret[np.arange(1, n), np.arange(n-1)] = c
    ret[np.arange(n-1), np.arange(1, n)] = e
    return ret


def rho(x):
    return np.sum(x * x, axis=1)  # diag(X*X')


def rayleigh_iteration_schur(gr, x0, niter=10, verbose=False):
    x = x0.copy()
    n, p = x.shape
    err = 100
    ii = 0
    tol = 1e-3
    while ii < niter and (err > tol):
        gr.calc_states(x)
        Lval = gr._state['Fx'] - np.dot(
            x, gr._state['RAYLEIGH'])
        err = np.linalg.norm(Lval)
        
        if err < tol:
            break
        if verbose:
            print Lval

        h_tensor = np.kron(x, np.eye(p))
        rhs = np.concatenate([
            gr._state['Fx'].reshape(-1, 1),
            h_tensor.reshape(n*p, -1)], axis=1)
        res_ = np.linalg.solve(
            gr._state['L_x'], rhs)
        
        j_c_zeta = np.dot(np.kron(x.T, np.eye(p)), res_[:, 1:])
        j_c_nu = np.dot(x.T, res_[:, 0].reshape(n, p)).reshape(p*p, -1)
        lbd_ = np.linalg.solve(j_c_zeta, j_c_nu)
        eta_ = np.dot(res_[:, 1:], lbd_).reshape(n, p) -\
            res_[:, 0].reshape(n, p)
        x = gr.retraction(x, eta_.reshape(n, p))
        ii += 1
    return x, gr._state['Fx'], gr._state['RAYLEIGH'], eta_


class grass_nonlinear_eigen(grassmann):
    """class for the nonlinear eigenspace problem below
    """
    def __init__(self, alpha, L, n, p):
        super(grass_nonlinear_eigen, self).__init__(
            (n, p), (p, p))
        self.alpha = alpha
        self.L = L

    def calc_states(self, x):
        n, p = x.shape
        self._x = x
        alpha = self.alpha
        rhoX = rho(x)
        self._state['rhoX'] = rhoX
        LX = np.dot(L, x)
        Linv = np.linalg.inv(L)
        LInvRhoX = np.dot(Linv, rhoX)
        fx = 0.5 * np.trace(np.dot(x.T, LX)) +\
            (alpha/4)*(np.dot(rhoX.T, LInvRhoX))
        Fx = np.dot(L, x) + alpha*np.dot(np.diag(LInvRhoX), x)
        self._state['fX'] = fx
        self._state['Fx'] = Fx
        self._state['LInvRhoX'] = LInvRhoX
        self._state['RAYLEIGH'] = np.dot(x.T, Fx)
        proj = np.eye(n) - np.dot(x, x.T)
        middle_term = np.zeros((n*p, n, p))
        for ii in range(n):
            for jj in range(p):
                l1 = np.dot(Linv, (x[ii, jj] * proj[:, ii]))
                middle_term[:, ii, jj] =\
                    (l1[:, None] * x).reshape(-1)
        self._state['J_F'] = np.kron(
            self.L + alpha * np.diag(LInvRhoX), np.eye(p))
        self._state['J_F'] += 2 * alpha * middle_term.reshape(n*p, n*p)
        self._state['L_x'] = self._state['J_F'] -\
            np.kron(np.eye(n), self._state['RAYLEIGH'].T)

    def J_F(self, U):
        U1 = U - np.dot(self._x, np.dot(self._x.T, U))
        rhoXdot = 2*np.sum(self._x * U1, axis=1)
        LinvRhoXdot = np.linalg.solve(L, rhoXdot)
        h = np.dot(L, U) + alpha * np.dot(np.diag(LinvRhoXdot), self._x)
        h += self.alpha * np.dot(np.diag(self._state['LInvRhoX']), U)
        return h


def nonlinear_eigenspace(L, p=10, alpha=1.):
    """minimize 0.5*trace(X'*L*X) + (alpha/4)*(rho(X)*L\(rho(X)))
    over X such that X'*X = Identity,

    where L is of size n-by-n,
    X is an n-by-k matrix, and
    rho(X) is the diagonal part of X*X'.

    This example is motivated in the paper
    "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
    Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin,
    SIAM Journal on Matrix Analysis and Applications, 36(2), 752-774, 2015.
    If no inputs are provided, generate a  discrete Laplacian operator.
    This is for illustration purposes only.
    The default example corresponds to Case (c) of Example 6.2 of the
     above referenced paper.
    """
    
    n = L.shape[0]
    assert L.shape[1] == n, 'L must be square.'
            
    x = np.random.randn(n, p)
    U, S, V = np.linalg.svd(x, 0)
    X = np.dot(U, V.T)
    L1 = L + alpha * np.diag(np.linalg.solve(L, rho(X)))
    e, v = np.linalg.eigh(L1)
    x0 = v[:, :p]
    gr = grass_nonlinear_eigen(alpha, L, n, p)
    res = rayleigh_iteration_schur(gr, x0, niter=10)
    print res

    
if __name__ == '__main__':
    n = 100
    p = 10
    
    n = 10
    p = 5
    L = gen_tridiag(n, -1, 2, -1)
    alpha = 1.
    nonlinear_eigenspace(L, p, alpha)
    
