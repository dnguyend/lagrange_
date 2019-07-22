from __future__ import print_function
import numpy as np
from scipy.linalg import solve
from Lagrangian import Lagrangian
from Lagrangian import state_requests as sr
from Lagrangian import state_keys as sk


class explicit_matrix_lagrangian(Lagrangian):
    """ Lagrangian where x is a vector
    and output is a vector
    constraint output is a vector
    H is a matrix
    L = F - H lbd
    """

    def calc_states(self, x, lbd, states=sr[sk.explicit_newton_raphson]):
        """
        """
        self.constraints.calc_states(x)
        for k in self.constraints.available_states:
            self._state[k] = self.constraints[k]
        self._state['F'] = self.F(x)
        self._state['H'] = self.calc_H(x)
        if 'HT' in states:
            if self.has_HT:
                self._state['HT'] = self.calc_HT(x)
                
        if 'HTH' in states:
            if 'HT' in self._state:
                self._state['HTH'] = np.dot(
                    self._state['HT'], self._state['H'])
            else:
                self._state['HTH'] = np.dot(
                    self._state['H'].T, self._state['H'])

        if 'J_C_H' in states:
            self._state['J_C_H'] = np.dot(
                self._state['J_C'], self._state['H'])

        if 'RAYLEIGH' in states:
            HF = np.dot(self._state['H'].T, self._state['F'])
            lbd = self.hth_solver(HF)
            self._state['RAYLEIGH'] = lbd
        self._state['L'] = self._state['F'] - np.dot(
            self._state['H'], lbd)
        self._state['J_F'] = self.calc_J_F(x)
        self._state['J_H'] = self.calc_J_H(x)
        self._state['LBD_J_H'] = np.tensordot(
            self._state['J_H'].T, lbd, [1, 0])
        
        self._state['L_x'] = self._state['J_F'] - self._state['LBD_J_H']
        
        if 'J_F2' in states:
            self._state['J_F2'] = self.calc_J_F2(x)

        if 'J_H2' in states:
            self._state['J_H2'] = self.calc_J_H2(x)

        if 'J_RAYLEIGH' in states:
            self._state['J_RAYLEIGH'] = self.calc_J_RAYLEIGH(x)

    def L_lambda(self, lbd):
        return np.dot(self._state['H'], lbd)

    def eigen_solver(self, b):
        return solve(self._state['L_x'], b)
                
    def hth_solver(self, b):
        """ solving a linear equation
        np.tensordot(a, y, 1) = b
        where a is of shape
        (shape_lagrange, shape_lagrange)
        and b is of shape shape_lagrange
        while we can do this by vectorizing
        it could be much simpler if we know more
        about the problem structure - so this should
        be override depending on problem
        if vector then it is just a vector solver
        if b is a matrix and a is a diagonal tensor
        then also a matrix solver
        """
        return solve(self._state['HTH'], b)

    def j_c_zeta_solver(self, zeta, rhs):
        j_c_zeta = np.dot(self._state['J_C'], zeta)
        if isinstance(rhs, list):
            return (solve(j_c_zeta, rhsa) for rhsa in rhs)
        else:
            return solve(j_c_zeta, rhs)

    def j_c_H_solver(self, rhs):
        # j_c_zeta = np.dot(self._state['J_C'], zeta)
        return solve(self._state['J_C_H'], rhs)

    def J_C(self, d_x):
        return np.dot(self._state['J_C'], d_x)

    def J_C2(self, d_x):
        """Evaluate the partial derivative of
        H with respect to lbd at x, lbd
        """
        return np.tensordot(
            np.tensordot(self._state['J_C2'], d_x, 1), d_x, 1)

    def J_F(self, d_x):
        return np.dot(self._state['J_C'], d_x)

    def j_F2(self, d_x):
        """Evaluate the second derivative of
        F with respect to lbd at x, lbd
        """
        return np.tensordot(
            np.tensordot(self._state['J_F2'], d_x, 1), d_x, 1)

    def J_H(self, d_x):
        """Evaluate the partial derivative of
        H with respect to lbd at x, lbd
        J_H if not override is a tensor
        of shape shape_constraint, shape_in, shape_in
        """
        return np.tensordot(
            self._state['J_H'], d_x, 1)
        
    def J_H2(self, d_x):
        """Evaluate the partial derivative of
        H with respect to lbd at x, lbd
        """
        return np.tensordot(
            np.tensordot(self._state['J_H2'], d_x, 1), d_x, 1)

    def J_RAYLEIGH(self, d_x):
        """Evaluate the partial derivative of
        H with respect to lbd at x, lbd
        """
        return np.tensordot(self._state['J_RAYLEIGH'], d_x, 1)

    def Pi_H(self, b):
        if 'HT' in self._state:
            hb = np.dot(self['HT'], b)
        else:
            hb = np.dot(self['H'].T, b)
        c = np.dot(self['H'], self.hth_solver(hb))
        return b - c

