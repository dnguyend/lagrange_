import numpy as np
import scipy.linalg


def symmetric_tv_mode_product(T, x, modes) -> np.ndarray:
    v = T
    for i in range(modes):
        v = np.tensordot(v, x, axes=1)
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
        x_init = x_init/np.linalg.norm(x_init)

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
        H = (m-1)*T_x_m_2-lbd*np.eye(n)
        H_p = U_x_k.T @ H @ U_x_k
        H_p_inv = np.linalg.inv(H_p)

        # fix eigenvector
        y = -U_x_k @ H_p_inv @ U_x_k.T @ g
        x_k_n = (x_k + y)/(np.linalg.norm(x_k + y))

        #  update residual and lbd
        R = np.linalg.norm(x_k-x_k_n)
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
        x_init = x_init/np.linalg.norm(x_init)

    # init lambda_(k) and x_(k)
    lbd = symmetric_tv_mode_product(T, x_init, m)
    x_k = x_init.copy()
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k
        rhs = np.concatenate(
            [x_k.reshape(-1, 1), T_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2-lbd*np.eye(n)
        lhs = np.linalg.solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum(x_k * lhs[:, 1]) / np.sum(x_k * lhs[:, 0])) - lhs[:, 1]
        x_k_n = (x_k + y)/(np.linalg.norm(x_k + y))

        #  update residual and lbd
        R = np.linalg.norm(x_k-x_k_n)
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
        x_init = x_init/np.linalg.norm(x_init)

    # init lambda_(k) and x_(k)
    lbd = symmetric_tv_mode_product(T, x_init, m)
    x_k = x_init.copy()
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        if do_chebyshev:
            T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
            T_x_m_2 = np.tensordot(T_x_m_3, x_k, axes=1)
        else:
            T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k
        rhs = np.concatenate(
            [x_k.reshape(-1, 1), T_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2-lbd*np.eye(n)
        lhs = np.linalg.solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum(x_k * lhs[:, 1]) / np.sum(x_k * lhs[:, 0])) - lhs[:, 1]
        if do_chebyshev and (np.linalg.norm(y) < 1e-2):
            J_R_eta = y.T @ T_x_m_1 + (m-1) * x_k.T @ T_x_m_2 @ y -\
                      2*(x_k.T @ y)
            L_x_lbd = -y * J_R_eta
            L_x_x = (m-1) * (m-2) * np.tensordot(T_x_m_3, y, axes=1) @ y
            T_a = np.linalg.solve(H, -L_x_lbd - 0.5 * L_x_x)
            T_adj = T_a - lhs[:, 0] * np.sum(x_k * T_a) / np.sum(
                x_k * lhs[:, 0])
            x_k_n = x_k + y + T_adj
            x_k_n /= np.linalg.norm(x_k_n)
        else:
            x_k_n = (x_k + y) / np.linalg.norm(x_k + y)
        
        # x_k_n = (x_k + y)/(np.linalg.norm(x_k + y))

        #  update residual and lbd
        R = np.linalg.norm(x_k-x_k_n)
        x_k = x_k_n
        lbd = symmetric_tv_mode_product(T, x_k, m)
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge


def schur_form_rayleigh_chebyshev_linear(
        T, max_itr, delta, x_init, u=None, do_chebyshev=True):
    """Schur form rayleigh to find complex
    eigen pair. We assume initial value is given. From
    there we decide if it a real or complex problem
    Constraint is u.T @ x = 1
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    n = T.shape[0]
    R = 1
    converge = False
    if u is None:
        u = np.zeros_like(x_init)
        u[0] += 1.

    # init lambda_(k) and x_(k)
    x_k = x_init.copy()
    ctr = 0
    if do_chebyshev:
        T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
        T_x_m_2 = np.tensordot(T_x_m_3, x_k, axes=1)
    else:
        T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
    T_x_m_1 = T_x_m_2 @ x_k
    lbd = u.T @ T_x_m_1 / (u.T @ x_init)
    
    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        rhs = np.concatenate(
            [x_k.reshape(-1, 1), T_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2-lbd*np.eye(n)
        lhs = np.linalg.solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum(u * lhs[:, 1]) / np.sum(u * lhs[:, 0])) - lhs[:, 1]
        if do_chebyshev and (np.linalg.norm(y) < 1e-2):
            J_R_eta = u.T @ (H @ y)
            L_x_lbd = -y * J_R_eta
            L_x_x = (m-1) * (m-2) * np.tensordot(T_x_m_3, y, axes=1) @ y
            T_a = np.linalg.solve(H, -L_x_lbd - 0.5 * L_x_x)
            T_adj = T_a - lhs[:, 0] * np.sum(u * T_a) / np.sum(u * lhs[:, 0])
            x_k_n = x_k + y + T_adj
            x_k_n /= (u @ x_k_n)
        else:
            x_k_n = (x_k + y)/(u.T @ (x_k + y))
        
        #  update residual and lbd
        R = np.linalg.norm(x_k-x_k_n)
        x_k = x_k_n
        if do_chebyshev:
            T_x_m_3 = symmetric_tv_mode_product(T, x_k, m-3)
            T_x_m_2 = np.tensordot(T_x_m_3, x_k, axes=1)
        else:
            T_x_m_2 = symmetric_tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k

        lbd = (u.T @ T_x_m_1) / (u.T @ x_k)
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge
