import numpy as np
import sys
import scipy.linalg
from numpy import concatenate, tensordot, eye, zeros, zeros_like,\
    power, sqrt, exp, pi

from numpy.linalg import solve, inv, norm

if sys.version_info[0] < 3:
    class SimpleNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

            def __repr__(self):
                keys = sorted(self.__dict__)
                items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
                return "{}({})".format(type(self).__name__, ", ".join(items))

            def __eq__(self, other):
                return self.__dict__ == other.__dict__
else:
    from types import SimpleNamespace


def symmetric_tv_mode_product(T, x, modes) -> np.ndarray:
    v = T
    for i in range(modes):
        v = tensordot(v, x, axes=1)
    return v


def orthogonal_newton_correction_method(
        T, max_itr, delta, x_init=None, use_cholesky=True):
    """function [x,lambda,ctr,converge] =
          orthogonal_newton_correction_method(T,max_itr,delta,x_init)

     Original matlab Code by Ariel Jaffe, Roi Weiss and Boaz Nadler
     2017, Weizmann Institute of Science
     ---------------------------------------------------------
    % DESCRIPTION:
    % 	This function implements the orthogonal Newton correction method
    %   for computing real eigenpairs of symmetric tensors
    %
    % Input:    T - A real and symmetric tensor
    %           max_itr - maximum number of iterations until
    %                     termination
    %           delta - convergence threshold
    %           x_init(opt) - initial point
    %
    % DEFAULT:
    %   if nargin<4 x_init is chosen randomly over the unit sphere
    %
    % Output:   x - output eigenvector
    %           ctr - number of iterations till convergence
    %           run_time - time till convergence
    %           convergence (1- converged)
    % ---------------------------------------------------------
    % FOR MORE DETAILS SEE:
    %   A. Jaffe, R. Weiss and B. Nadler.
    %   Newton correction methods for computing
    %   real eigenpairs of symmetric tensors (2017)
    % ---------------------------------------------------------
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    n = T.shape[0]
    R = 1
    converge = False

    # if not given as input, randomly initialize
    if x_init is None:
        x_init = np.random.randn(n)
        x_init = x_init/norm(x_init)

    # init lambda_(k) and x_(k)
    lbd = symmetric_tv_mode_product(T, x_init, m)
    x_k = x_init.copy()
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k
        g = -lbd * x_k + T_x_m_1

        # compute Hessian H(x_k) and projecected Hessian H_p(x_k)
        U_x_k = scipy.linalg.null_space(x_k.reshape(-1, 1).T)
        H = (m-1)*T_x_m_2-lbd*eye(n)
        H_p = U_x_k.T @ H @ U_x_k
        H_p_inv = inv(H_p)

        # fix eigenvector
        y = -U_x_k @ H_p_inv @ U_x_k.T @ g
        x_k_n = (x_k + y)/(norm(x_k + y))

        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n
        lbd = symmetric_tv_mode_product(T, x_k, m)
        # print("ctr=%d lbd=%f" % (ctr, lbd))
        ctr += 1

    x = x_k
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge


def schur_form_rayleigh(
        T, max_itr, delta, x_init=None):
    """Schur form rayleigh
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    n = T.shape[0]
    R = 1
    converge = False

    # if not given as input, randomly initialize
    if x_init is None:
        x_init = np.random.randn(n)
        x_init = x_init/norm(x_init)

    # init lambda_(k) and x_(k)
    lbd = symmetric_tv_mode_product(T, x_init, m)
    x_k = x_init.copy()
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k
        rhs = concatenate(
            [x_k.reshape(-1, 1), T_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2-lbd*eye(n)
        lhs = solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum(x_k * lhs[:, 1]) / np.sum(x_k * lhs[:, 0])) - lhs[:, 1]
        x_k_n = (x_k + y)/(norm(x_k + y))

        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n
        lbd = symmetric_tv_mode_product(T, x_k, m)
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge


def schur_form_rayleigh_chebyshev(
        T, max_itr, delta, x_init=None, do_chebyshev=True):
    """Schur form rayleigh
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    n = T.shape[0]
    R = 1
    converge = False

    # if not given as input, randomly initialize
    if x_init is None:
        x_init = np.random.randn(n)
        x_init = x_init/norm(x_init)

    # init lambda_(k) and x_(k)

    x_k = x_init.copy()
    if do_chebyshev:
        T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
        T_x_m_2 = tensordot(T_x_m_3, x_k, axes=1)
    else:
        T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
    T_x_m_1 = T_x_m_2 @ x_k
    lbd = x_k.T @ T_x_m_1
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        rhs = concatenate(
            [x_k.reshape(-1, 1), T_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2-lbd*eye(n)
        lhs = solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum(x_k * lhs[:, 1]) / np.sum(x_k * lhs[:, 0])) - lhs[:, 1]
        if do_chebyshev and (norm(y) < 50e-2):
            J_R_eta = y.T @ T_x_m_1 + (m-1) * x_k.T @ T_x_m_2 @ y -\
                      2*(x_k.T @ y) * lbd
            L_x_lbd = -y * J_R_eta
            L_x_x = (m-1) * (m-2) * tensordot(T_x_m_3, y, axes=1) @ y
            T_a = solve(H, -L_x_lbd - 0.5 * L_x_x)
            T_adj = T_a - lhs[:, 0] * np.sum(x_k * T_a) / np.sum(
                x_k * lhs[:, 0])
            x_k_n = x_k + y + T_adj
            x_k_n /= norm(x_k_n)
        else:
            x_k_n = (x_k + y) / norm(x_k + y)

        # x_k_n = (x_k + y)/(np.linalg.norm(x_k + y))

        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n
        if do_chebyshev:
            T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
            T_x_m_2 = tensordot(T_x_m_3, x_k, axes=1)
        else:
            T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k

        lbd = x_k.T @ T_x_m_1
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k
    err = norm(symmetric_tv_mode_product(
        T, x, m-1) - lbd * x)

    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge, err


def schur_form_rayleigh_chebyshev_unitary(
        T, max_itr, delta, x_init=None, do_chebyshev=True):
    """Schur form rayleigh chebyshev unitary
    T and x are complex. Constraint is x^H x = 1
    lbd is real
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    n = T.shape[0]
    R = 1
    converge = False

    # if not given as input, randomly initialize
    if x_init is None:
        x_init = np.random.randn(n)
        x_init = x_init/norm(x_init)

    # init lambda_(k) and x_(k)
    x_k = x_init.copy()
    if do_chebyshev:
        T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
        T_x_m_2 = tensordot(T_x_m_3, x_k, axes=1)

    else:
        T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
    T_x_m_1 = T_x_m_2 @ x_k

    lbd = (x_k.conjugate().T @ T_x_m_1).real
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        rhs = concatenate(
            [x_k.reshape(-1, 1), T_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2-lbd*eye(n)
        lhs = solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum((x_k.conjugate() * lhs[:, 1]).real) /
            np.sum((x_k.conjugate() * lhs[:, 0]).real)) - lhs[:, 1]
        if do_chebyshev and (np.linalg.norm(y) < 30e-2):
            J_R_eta = y.conjugate().T @ T_x_m_1 +\
                      (m-1) * x_k.conjugate().T @ T_x_m_2 @ y -\
                      2*(x_k.conjugate().T @ y)*lbd
            L_x_lbd = -y * J_R_eta
            L_x_x = (m-1) * (m-2) * np.tensordot(T_x_m_3, y, axes=1) @ y
            T_a = np.linalg.solve(H, -L_x_lbd - 0.5 * L_x_x)
            T_adj = T_a - lhs[:, 0] *\
                np.sum((x_k.conjugate() * T_a).real) /\
                np.sum((x_k.conjugate() * lhs[:, 0]).real)
            x_k_n = x_k + y + T_adj
            x_k_n /= norm(x_k_n)
        else:
            x_k_n = (x_k + y) / norm(x_k + y)

        # x_k_n = (x_k + y)/(np.linalg.norm(x_k + y))

        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n
        if do_chebyshev:
            T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
            T_x_m_2 = np.tensordot(T_x_m_3, x_k, axes=1)
        else:
            T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k

        lbd = (x_k.conjugate().T @ T_x_m_1).real
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k
    err = norm(symmetric_tv_mode_product(
        T, x, m-1) - lbd * x)
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge, err


def _schur_form_rayleigh_chebyshev_linear(
        T, max_itr, delta, x_init, u=None):
    """Schur form rayleigh to find complex
    eigen pair. We assume initial value is given. From
    there we decide if it a real or complex problem
    Constraint is u.T @ x = 1
    for this, the function should be
    T(I, x) - lbd * (x.T @ x)^{(m-1)/2} x
    """
    do_chebyshev = False
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    n = T.shape[0]
    R = 1
    converge = False
    if u is None:
        u = zeros_like(x_init)
        u[0] += 1.

    # init lambda_(k) and x_(k)
    x_k = x_init / (u.T @ x_init)
    ctr = 0
    if do_chebyshev:
        T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
        T_x_m_2 = tensordot(T_x_m_3, x_k, axes=1)
    else:
        T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
    T_x_m_1 = T_x_m_2 @ x_k
    x_t_x = x_k.T @ x_k
    x_t_x_m_1_2 = power(x_t_x, .5*(m-1))
    lbd = u.T @ T_x_m_1 / (u.T @ x_k) / x_t_x_m_1_2
    
    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        rhs = concatenate(
            [x_t_x_m_1_2 * x_k.reshape(-1, 1), T_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2 - lbd*x_t_x_m_1_2*eye(n) -\
            (m-1)*lbd*x_t_x_m_1_2/x_t_x*(x_k.reshape(-1, 1) @
                                         x_k.reshape(1, -1))
        lhs = solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum(u * lhs[:, 1]) / np.sum(u * lhs[:, 0])) - lhs[:, 1]
        if do_chebyshev and (norm(y) < 1e-2):
            J_R_eta = u.T @ (H @ y)
            L_x_lbd = -y * J_R_eta
            L_x_x = (m-1) * (m-2) * tensordot(T_x_m_3, y, axes=1) @ y
            T_a = solve(H, -L_x_lbd - 0.5 * L_x_x)
            T_adj = T_a - lhs[:, 0] * np.sum(u * T_a) / np.sum(u * lhs[:, 0])
            x_k_n = x_k + y + T_adj
            x_k_n /= (u @ x_k_n)
        else:
            x_k_n = (x_k + y)/(u.T @ (x_k + y))
        
        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n
        if do_chebyshev:
            T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
            T_x_m_2 = tensordot(T_x_m_3, x_k, axes=1)
        else:
            T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k
        x_t_x = x_k.T @ x_k
        x_t_x_m_1_2 = power(x_t_x, .5*(m-1))
        lbd = u.T @ T_x_m_1 / (u.T @ x_k) / x_t_x_m_1_2
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k / (x_k.T / x_k)
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge


def schur_form_rayleigh_linear(
        T, max_itr, delta, x_init, u=None):
    """Schur form rayleigh to find complex
    eigen pair. We assume initial value is given. From
    there we decide if it a real or complex problem
    Constraint is u.T @ x = 1
    for this, the function should be
    T(I, x) - lbd * (x.T @ x)^{(m-1)/2} x
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    n = T.shape[0]
    R = 1
    converge = False
    if u is None:
        u = zeros_like(x_init)
        u[0] += 1.

    # init lambda_(k) and x_(k)
    x_k = x_init / (u.T @ x_init)
    ctr = 0
    T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
    T_x_m_1 = T_x_m_2 @ x_k
    x_t_x = x_k.T @ x_k
    x_t_x_m_2_2 = power(x_t_x, .5*(m-2))
    lbd = u.T @ T_x_m_1 / (u.T @ x_k) / x_t_x_m_2_2
    
    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        rhs = concatenate(
            [x_t_x_m_2_2 * x_k.reshape(-1, 1), T_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2 - lbd*x_t_x_m_2_2*eye(n) -\
            (m-2)*lbd*x_t_x_m_2_2/x_t_x*(x_k.reshape(-1, 1) @
                                         x_k.reshape(1, -1))
        lhs = solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum(u * lhs[:, 1]) / np.sum(u * lhs[:, 0])) - lhs[:, 1]
        x_k_n = (x_k + y)/(u.T @ (x_k + y))
        
        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n
        T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k
        x_t_x = x_k.T @ x_k
        x_t_x_m_2_2 = power(x_t_x, .5*(m-2))
        lbd = u.T @ T_x_m_1 / (u.T @ x_k) / x_t_x_m_2_2
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k / norm(x_k)
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge


def complex_eigen_cnt(n, m):
    if m == 2:
        return n
    return (power(m-1, n)-1) // (m-2)


def find_eig_cnt(all_eig):
    first_nan = np.where(np.isnan(all_eig.x))[0]
    if first_nan.shape[0] == 0:
        return None
    else:
        return first_nan[0]

    
def normalize_real_positive(lbd, x, m, tol):
    """ First try to make it to a real pair
    if not possible. If not then make lambda real
    return is_self_conj, is_real, new_lbd, new_x
    """
    u = (sqrt(x @ x)).conjugate()
    is_self_conj = norm(x.conjugate() - u*u*x) < tol
    new_x = x * u
    if np.sum(new_x.imag) < tol:
        # try to flip. if u **(m-2) > 0 use it:
        lbd_factor = u**(m-2)
        if np.abs(lbd_factor*lbd_factor - 1) < tol:
            lbd_factor = lbd_factor.real
            if lbd * lbd_factor > 0:
                return is_self_conj, True, lbd * lbd_factor, new_x
            elif m % 2 == 1:
                return is_self_conj, True, -lbd * lbd_factor, -new_x
            else:
                return is_self_conj, True, lbd * lbd_factor, new_x
    if lbd < 0:
        return is_self_conj, False, -lbd, x * exp(pi/(m-2)*1j)
    else:
        return is_self_conj, False, lbd, x


def _insert_eigen(all_eig, x, lbd, eig_cnt, m, tol):
    """
    force eigen values to be positive if possible
    if x is not similar to a vector in all_eig.x
    then:
       insert pair x, conj(x) if x is not self conjugate
       otherwise insert x
    all_eig has a structure: lbd, x, is_self_conj, is_real
    """
    is_self_conj, is_real, norm_lbd, norm_x = normalize_real_positive(
        lbd, x, m, tol)

    if is_self_conj:
        good_x = [norm_x]
    else:
        good_x = [norm_x, norm_x.conjugate()]

    factors = all_eig.x[:eig_cnt, :] @ norm_x.conjugate()
    fidx = np.where(np.abs(factors ** (m-2) - 1) < tol)[0]
    if fidx.shape[0]:
        all_diffs = all_eig.x[:eig_cnt, :][fidx, :] -\
                    factors[fidx][:, None] * norm_x[None, :]
        if np.where(np.sum(np.abs(all_diffs)) < tol * x.shape[0])[0].shape[0]:
            return eig_cnt

    for j in range(len(good_x)):
        all_eig.lbd[eig_cnt+j] = norm_lbd
        all_eig.x[eig_cnt+j] = good_x[j]
        all_eig.is_self_conj[eig_cnt+j] = is_self_conj
        all_eig.is_real[eig_cnt+j] = is_real

    return eig_cnt + len(good_x)


def find_all_unitary_eigenpair(
        all_eig, eig_cnt, A, max_itr, max_test=int(1e6), tol=1e-10):
    """ output is the table of results
     2n*+2 columns: lbd, is self conjugate, x_real, x_imag
    This is the raw version, since the output vector x
    is not yet normalized to be real when possible
    """
    n = A.shape[0]
    m = len(A.shape)
    n_eig = complex_eigen_cnt(n, m)
    if all_eig is None:
        all_eig = SimpleNamespace(
            lbd=np.full((n_eig), np.nan, dtype=float),
            x=np.full((n_eig, n), np.nan, dtype=np.complex),
            is_self_conj=zeros((n_eig), dtype=bool),
            is_real=zeros((n_eig), dtype=bool))
        eig_cnt = 0
    elif eig_cnt is None:
        eig_cnt = find_eig_cnt(all_eig)
        if eig_cnt is None:
            return all_eig

    for jj in range(max_test):
        x0r = np.random.randn(2*n)
        x0r /= norm(x0r)
        x0 = x0r[:n] + x0r[n:] * 1.j
        # if there are odd numbers left,
        # try to find a real root
        draw = np.random.uniform(0, 1, 1)
        # 50% try real root
        if (draw < .5) and ((n_eig - eig_cnt) % 2 == 1):
            try:
                x_r, lbd, ctr, converge, err = schur_form_rayleigh_chebyshev(
                    A, max_itr, tol, x_init=x0.real, do_chebyshev=False)
                x = x_r + 1j * zeros((x_r.shape[0]))
            except Exception as e:
                print(e)
                continue
        else:
            try:
                x, lbd, ctr, converge, err =\
                    schur_form_rayleigh_chebyshev_unitary(
                        A, max_itr, tol, x_init=x0, do_chebyshev=False)
            except Exception as e:
                print(e)
                continue
        old_eig = eig_cnt
        if converge and (err < tol):
            eig_cnt = _insert_eigen(all_eig, x, lbd, eig_cnt, m, tol)
        if eig_cnt == n_eig:
            break
        elif (eig_cnt > old_eig) and (eig_cnt % 10 == 0):
            print('Found %d eigenpairs' % eig_cnt)
    return all_eig, jj


