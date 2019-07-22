import numpy as np
from Lagrangian import state_keys as sk
from Lagrangian import state_requests as sr
_MAX_ERR = 1e-5
_MAX_ITER = 100


def explicit_newton_raphson(
        lagrangian, x0, lbd0,
        max_err=_MAX_ERR, max_iter=_MAX_ITER, feasible=True, verbose=False):
    """Implement newton raphson for explicit L
    """
    assert(lagrangian.F is not None)
    assert(lagrangian.L_lambda is not None)

    zeta = np.array([0.5 / max_err])
    MAX_ZETA = 1. / max_err
    i = 0
    x = x0.copy()
    lbd = lbd0
    err = max_err + 1
    while (np.linalg.norm(zeta) < MAX_ZETA) and\
          (i < max_iter) and (err > max_err):
        lagrangian.calc_states(
            x, lbd, states=sr[sk.explicit_newton_raphson])

        if err < max_err:
            break
        zeta = lagrangian['Lx_inv_H']
        nu = lagrangian.eigen_solver(
            lagrangian['F'])

        j_c_nu = lagrangian.J_C(nu)
        if feasible:
            lbd = lagrangian.j_c_zeta_solver(
                zeta, j_c_nu)
        else:
            lbd = lagrangian.j_c_zeta_solver(
                zeta, j_c_nu - lagrangian['C'])

        eta = -nu + lagrangian.tensordot(zeta, lbd)
        if verbose:
            print 'i= %d' % i
            print "err_bfore = %s" % str(lagrangian.L(x + eta, lbd))

        if feasible:
            xf = lagrangian.constraints.retraction(x, eta)
            if xf is not None:
                x = xf
        if (not feasible) or (xf is None):
            x += eta

        if verbose:
            print "zeta = %s" % str(zeta)
            print "eta = %s" % str(eta)
            print "x=%s lbd=%s" % (str(x), str(lbd))
            print "err = %s" % str(lagrangian.L(x, lbd))
            if feasible and (xf is None):
                print "cannot reach constraint use infeasible"

        i += 1        
    return {'x': x, 'lbd': lbd,
            'n_iter': i, 'err': lagrangian.L(x, lbd)}


def explicit_chebyshev(
        lagrangian, x0, lbd0,
        max_err=_MAX_ERR, max_iter=_MAX_ITER, feasible=True, verbose=False):
    """Implement newton raphson for explicit L
    """
    assert(lagrangian.F is not None)
    assert(lagrangian.L_lambda is not None)
    # eta = np.array([_MAX_ERR + 1])
    zeta = np.array([0.5 / max_err])
    MAX_ZETA = 1. / max_err

    i = 0
    x = x0.copy()
    lbd = lbd0

    while (np.linalg.norm(zeta) < MAX_ZETA) and (i < max_iter):
        lagrangian.calc_states(
            x, lbd, states=sr[sk.explicit_chebyshev])
        zeta = lagrangian['Lx_inv_H']
        nu = lagrangian.eigen_solver(
            lagrangian['F'])

        j_c_nu = lagrangian.J_C(nu)
        old_lbd = lbd
        if feasible:
            lbd = lagrangian.j_c_zeta_solver(
                zeta, j_c_nu)
        else:
            lbd = lagrangian.j_c_zeta_solver(
                zeta, j_c_nu - lagrangian['C'])
        delta = lbd - old_lbd
        old_lbd = lbd
        
        eta = -nu + lagrangian.tensordot(zeta, lbd)
        # evaluate J2 Lhat:
        JL2 = lagrangian.J_F2(eta)
        """
        JL2 -= np.tensordot(
            lagrangian.J_H2(eta), old_lbd, len(old_lbd.shape))
        """
        JL2 -= lagrangian.J_H2(eta, old_lbd)
        
        JL2 -= 2 * lagrangian.J_H(eta, delta)
        jj_c_eta = lagrangian.J_C2(eta)

        # now evaluate (JLhat)^{-1} J2 Lhat
        invLX_JL2 = lagrangian.eigen_solver(JL2)

        # p1 = lagrangian.j_c_zeta_solver(jj_c_eta)
        
        delta2 = lagrangian.j_c_zeta_solver(
            zeta, lagrangian.J_C(invLX_JL2) - jj_c_eta)
        
        eta2 = invLX_JL2 + lagrangian.tensordot(
            zeta, delta2)

        lbd = lbd - 0.5 * delta2
        eta_tot = eta - 0.5 * eta2
        if feasible:
            xf = lagrangian.constraints.retraction(x, eta_tot)
            if xf is not None:
                x = xf
        if (not feasible) or (xf is None):
            x += eta_tot

        if verbose:
            print "i=%d" % i
            print "zeta = %s" % str(zeta)
            print "eta = %s" % str(eta)
            print "x=%s lbd=%s" % (str(x), str(lbd))
            print "err = %s" % str(lagrangian.L(x, lbd))
            if feasible and (xf is None):
                print "cannot reach constraint use infeasible"

        i += 1        
    return {'x': x, 'lbd': lbd,
            'n_iter': i, 'err': lagrangian.L(x, lbd)}


def implicit_newton_raphson(
        lagrangian, x0, lbd0,
        max_err=_MAX_ERR, max_iter=_MAX_ITER, feasible=True, verbose=False):

    """Implement implicit newton raphson
    """
    zeta = np.array([0.5 / max_err])
    MAX_ZETA = 1. / max_err
    i = 0
    x = x0.copy()
    lbd = lbd0

    while (np.linalg.norm(zeta) < MAX_ZETA) and (i < max_iter):
        lagrangian.calc_states(x, lbd)
        zeta = lagrangian['Lx_inv_L_lambda']
        nu = lagrangian.eigen_solver(
            lagrangian['L'])

        j_c_nu = lagrangian.J_C(nu)

        if feasible:
            delta = -lagrangian.j_c_zeta_solver(
                zeta, j_c_nu)

        else:
            pt1, pt2 = lagrangian.j_c_zeta_solver(
                zeta, [j_c_nu, lagrangian['C']])
            delta = pt2 - pt1

        eta = -nu - lagrangian.tensordot(zeta, delta)
        lbd = lbd + delta
        if feasible:
            xf = lagrangian.constraints.retraction(x, eta)
            if xf is not None:
                x = xf
        if (not feasible) or (xf is None):
            x += eta

        if verbose:
            print "i=%d" % i
            print "zeta = %s" % str(zeta)
            print "eta = %s" % str(eta)
            print "x=%s lbd=%s" % (str(x), str(lbd))
            print "err = %s" % str(lagrangian.L(x, lbd))
            if feasible and (xf is None):
                print "cannot reach constraint use infeasible"
        i += 1
    return {'x': x, 'lbd': lbd,
            'n_iter': i, 'err': lagrangian.L(x, lbd)}


def rayleigh_quotient_iteration(
        lagrangian, x0, tangent_rayleigh=True,
        max_err=_MAX_ERR, max_iter=_MAX_ITER, verbose=False):
    """Implement rayleigh iteration
    """
    assert(lagrangian.F is not None)
    assert(lagrangian.L_lambda is not None)
    zeta = np.array([0.5 / max_err])
    MAX_ZETA = 1 / max_err

    i = 0
    x = x0.copy()

    while (np.linalg.norm(zeta) < MAX_ZETA) and (i < max_iter):
        lagrangian.calc_states(x, lbd=None, states=sr[sk.rayleigh])
        zeta = lagrangian['Lx_inv_H']
        nu = lagrangian.eigen_solver(
            lagrangian['F'])
        j_c_nu = lagrangian.J_C(nu)
        if tangent_rayleigh:
            lbd2 = lagrangian.j_c_zeta_solver(zeta, j_c_nu)
            eta = lagrangian.tensordot(zeta, lbd2) - nu
        else:
            # lbd2 = lagrangian.j_c_zeta_solver(zeta, j_c_nu)
            eta = lagrangian.tensordot(zeta, lagrangian['RAYLEIGH']) - nu
        
        x = lagrangian.constraints.retraction(x, eta)
        if verbose:
            print 'i=%d' % i
            print "zeta = %s" % str(zeta)
            print "eta = %s" % str(eta)
            print "x=%s lbd=%s" % (str(x), str(lagrangian['RAYLEIGH']))
            print "err = %s" % str(lagrangian.L(x, lagrangian['RAYLEIGH']))
        i += 1
    lbd = lagrangian['RAYLEIGH']
    return {'x': x, 'lbd': lbd,
            'n_iter': i, 'err': lagrangian.L(x, lbd)}


def rayleigh_chebyshev(lagrangian, x0,
                       max_err=_MAX_ERR, max_iter=_MAX_ITER, verbose=False):

    """Implement rayleigh chebyshev iteration
    """
    assert(lagrangian.F is not None)
    assert(lagrangian.L_lambda is not None)

    zeta = np.array([0.5 / max_err])
    MAX_ZETA = 1 / max_err

    i = 0
    x = x0.copy()

    while (np.linalg.norm(zeta) < MAX_ZETA) and (i < max_iter):
        lagrangian.calc_states(x, lbd=None, states=sr[sk.rayleigh_chebyshev])
        zeta = lagrangian['Lx_inv_H']
        nu = lagrangian.eigen_solver(
            lagrangian['F'])

        lbd = lagrangian['RAYLEIGH']
 
        eta = -nu + lagrangian.tensordot(zeta, lbd)
        rhs_for_t = 0.5 * lagrangian.J_H2(eta, lbd)
        rhs_for_t -= 0.5 * lagrangian.J_F2(eta)
        j_r_nu = lagrangian.J_RAYLEIGH(eta)
        """
        rhs_for_t += np.tensordot(
            lagrangian.J_H(eta), j_r_nu, len(j_r_nu.shape))
        """
        rhs_for_t += lagrangian.J_H(eta, j_r_nu)

        T_ = lagrangian.eigen_solver(rhs_for_t)
        # tau_ = T_ - nu
        tau_ = T_ + eta

        j_c_t_ = lagrangian.J_C(tau_)
        j_c_zeta_m_j_c_t = lagrangian.j_c_zeta_solver(
            zeta, j_c_t_)

        tau = tau_ - lagrangian.tensordot(
            zeta, j_c_zeta_m_j_c_t)

        x = lagrangian.constraints.retraction(x, tau)
        if verbose:
            print 'i=%d' % i
            print "zeta = %s" % str(zeta)
            print "tau = %s" % str(tau)
            print "x=%s lbd=%s" % (str(x), str(lagrangian['RAYLEIGH']))
            print "err = %s" % str(lagrangian.L(x, lagrangian['RAYLEIGH']))
        i += 1
    lbd = lagrangian['RAYLEIGH']
    return {'x': x, 'lbd': lbd,
            'n_iter': i, 'err': lagrangian.L(x, lbd)}


