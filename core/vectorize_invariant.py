import numpy as np

A = None
n = 7
p = 2


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
