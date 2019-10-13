from __future__ import print_function

import numpy as np

k = 2
n = 8
# np.random.seed(0)
C = np.random.randn(k, n)
b = np.random.randn(k)

CCT = np.dot(C, C.T)
CCTinv = np.linalg.inv(CCT)
CTCCTinv = np.dot(C.T, CCTinv)
CCTinvC = np.dot(CCTinv, C)

a = np.diag(np.random.randint(1, 10, n) * .2 + 1.)
d = np.random.randint(-5, 5, n) * 1.


def F(x):
    # return np.dot(a, x + 5 * np.sin(3 * x)) + d
    return np.dot(a, x) + d


def J_F(x):
    # return np.dot(a, np.diag(1. + 15 * np.cos(3 * x)))
    return a

"""
def J_F2(x):
    ret = np.zeros((n, n, n), dtype=float)
    for i in range(n):
        ret[:, :, i] = np.diag
"""


def project(x):
    return x + np.dot(CTCCTinv, b - np.dot(C, x))


if __name__ == '__main__':
    x0 = np.random.randn(n)
    x = project(x0)
    
    for i in range(20):
        FX = F(x)
        J_FX = J_F(x)
        rayleigh = np.dot(CCTinvC, FX)
        zeta = np.linalg.solve(J_FX, C.T)
        nu = np.linalg.solve(J_FX, FX)
        eta = np.dot(zeta, rayleigh) - nu
        x = project(x + eta)
        print(x)
        print(np.dot(C, x) - b)
        print(F(x) - np.dot(C.T, rayleigh))

    CTCCTinvC = np.dot(C.T, CCTinvC)
    P = np.eye(n) - CTCCTinvC
    Pd = np.dot(P, d)

    AinvP = np.linalg.solve(a, P)
    ImPAinvP = np.dot(CTCCTinvC, AinvP)
    AinvPd = np.dot(AinvP, d)
    CTCCTinvb = np.dot(CTCCTinv, b)
    PAinvPd = np.dot(P, AinvPd)
    PA = np.dot(P, a)
    AinvPA = np.linalg.solve(a, PA)
    PAinvPA = np.dot(P, AinvPA) 
    PAinvPA_P = PAinvPA - P
    
    x1 = project(x0)
    for i in range(20):
        FX1 = F(x1)
        rayleigh = np.dot(CCTinvC, FX1)
        # x1_ = x1 - np.dot(AinvP, FX1)
        x1 = -np.dot(PAinvPA_P, x1) - PAinvPd + CTCCTinvb
        # x1 = project(x1_)
        # x1 = np.dot(CTCCTinv, b) + np.dot(P, x1_)
        # x1 = np.dot(CTCCTinv, b) - np.dot(CTCCTinvC, x1) + np.dot(ImPAinvP, FX1)
        print(x1)
        print(np.dot(C, x1) - b)
        print(F(x1) - np.dot(C.T, rayleigh))

    x2 = project(x0)
    for i in range(20):
        FX = F(x2)
        J_FX = J_F(x2)
        rayleigh = np.dot(CCTinvC, FX)
        zeta = np.linalg.solve(J_FX, C.T)
        nu = np.linalg.solve(J_FX, FX)

        lbd2 = np.linalg.solve(np.dot(C, zeta), np.dot(C, nu))
        eta = np.dot(zeta, lbd2) - nu
        x2 = project(x2 + eta)
        print(x2)
        print(np.dot(C, x2) - b)
        print(F(x2) - np.dot(C.T, rayleigh))

    # direct result        
    l1 = np.dot(C, np.linalg.solve(a, C.T))
    l2 = np.dot(C, np.linalg.solve(a, d))
    ll = np.linalg.solve(l1, l2 + b)  # lambda
    xdirect = np.linalg.solve(a, np.dot(C.T, ll) - d)
    print(xdirect)

    # Iterative results
    xx3 = CTCCTinvb - PAinvPd
    xx4 = CTCCTinvC + PAinvPA
    print(np.linalg.solve(xx4, xx3))



    
