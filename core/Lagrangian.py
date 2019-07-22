import abc
import numpy as np
from enum import Enum


class state_keys(Enum):
    """ state request keys
    """
    explicit_newton_raphson = 0
    implicit_newton_raphson = 1
    explicit_chebyshev = 2
    rayleigh = 3
    rayleigh_chebyshev = 4


state_requests = {
    state_keys.explicit_newton_raphson: [
        'F', 'H', 'L', 'J_F', 'J_H', 'LBD_J_H',
        'L_x'],
    state_keys.implicit_newton_raphson: ['L', 'L_x', 'L_lbd'],
    state_keys.explicit_chebyshev: [
        'F', 'H', 'L', 'J_F', 'J_H', 'LBD_J_H', 'L_x',
        'J_F2', 'J_H2'],
    state_keys.rayleigh: ['HTH', 'J_C_H', 'RAYLEIGH'],
    state_keys.rayleigh_chebyshev: [
        'HTH', 'J_C_H', 'RAYLEIGH', 'J_H2', 'J_F2', 'J_RAYLEIGH']
}    


class Lagrangian():
    """ a problem of type
    L(x, lbd) = 0
    or L(x, lbd, mu) = 0
    """

    __metaclass__ = abc.ABCMeta
    # we dont need to set the shapes
    # but useful to check

    _shape_in = None  # shape of x
    _shape_out = None  # shape of F
    _shape_constraint = None  # shape of constraints
    
    _constraints = None
    _args = None
    _state = {}
    _state_keys = ['J_C', 'L', 'L_lambda', 'C',
                   'F', 'H', 'Lx_inv', 'LBD_H',
                   'RAYLEIGH', 'J_F', 'J_F2',
                   'J_H', 'J_H2',
                   'J_c zeta', 'HTH',
                   'J_C_H']
    is_explicit = True
    has_HT = False
    
    def __getitem__(self, key):
        return self._state[key]
    
    @property
    def shape_in(self):
        return self._shape_in

    @property
    def shape_out(self):
        return self._shape_out

    @property
    def shape_constraint(self):
        return self._shape_constraint

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, in_args):
        self.args = in_args

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, in_constraints):
        self._constraints = in_constraints
        self._shape_constraint = in_constraints.shape_constraint

    def tensordot(self, zeta, eta):
        """ ususually need to override
        """
        return np.tensordot(zeta, eta, len(eta.shape))

    @abc.abstractmethod
    def F(self, x):
        """ x is an array of shape shape_in
        return array of shape shape_out
        """

    @abc.abstractmethod
    def calc_H(self, x):
        """ x is an array of shape shape_in
        return array of shape shape_constraint
        """

    def L(self, x, lbd):
        """Evaluate the lagrangean at x, lbd
        x of shape shape_in
        lbd of shape shape_constraint
        output shape (shape_out)
        """
        if self.is_explicit:
            g = self.F(x) + self.L_lambda(lbd)
            return g
        else:
            raise(ValueError("not implemented"))

    @abc.abstractmethod
    def L_lambda(self, d_lbd):
        """ partial derivatives with respect
        to lbd for implicit mode.
        To compute H(x) lbd we also use this function
        """
    
    def calc_states(self, x, lbd, states=None,
                    target=state_keys.explicit_newton_raphson):
        """Compute the state according to x and lbd
        depending on iteration method
        required state could be different
        the reason is higher derivatives
        while could be provided as tensor
        most of the time its not efficient
        a small part is need to compute the derivatives
        these small parts are precomuted and saved as states
        depending on iteration method the following
        should be saved in calc states
        L
        L_lambda
        C
        F
        H
        
        Structure precomputed for the following maps
        (states should cover these)
        J_C
        Lx^{-1}
        Rayleigh

        For Rayleigh Chebyshev
        J^2F
        J_H
        J_H2


        Other inverse to compute:
        Case Newton:
        J_c zeta: will be (shape_constraint, shape_constraint) tensor inverse.

        Case Rayleigh:
        H^TH inverse: (shape_constraint, shape_constraint) inverse.
        (J_C, H) inverse
        """

    @abc.abstractmethod
    def J_C(self, d_x):
        """ provide evaluation of J_C using states
        on d_x
        """

    @abc.abstractmethod
    def J_F(self, d_x):
        """Evaluate the partial derivative of the
        F with respect to lbd at x, lbd
        A tensor of shape (shape_out, shape_in)
        """

    @abc.abstractmethod
    def J_F2(self, d_x):
        """Evaluate the second derivative of
        F with respect to lbd at x, lbd
        """

    @abc.abstractmethod
    def J_H(self, d_x, d_lbd):
        """Evaluate the partial derivative of
        H with respect to lbd at x, lbd
        """

    @abc.abstractmethod
    def J_H2(self, d_x, d_lbd):
        """Evaluate the partial derivative of
        H with respect to lbd at x, lbd
        """

    @abc.abstractmethod
    def J_RAYLEIGH(self, d_x):
        """Evaluate Rayleigh on d_x
        """
        
    @abc.abstractmethod
    def eigen_solver(self, b):
        """Solve for eta in equation
        L_x eta = b
        assume jac_x and b are already computed and shaped to
        the correct matrix shape. Override for
        appropriate problems
        """

    @abc.abstractmethod
    def hth_solver(self, b):
        """ solving a linear equation
        np.tensordot(a, y, len(shape(y))) = b
        where a is of shape
        (shape_lagrange, shape_lagrange)
        and b is of shape shape_lagrange
        while we can do this by vectorizing
        it could be much simpler if we know more
        about the problem structure
        """

    @abc.abstractmethod
    def j_c_zeta_solver(self, zeta, rhs):
        """ solving J_C_zeta y = rhs
        """

    @abc.abstractmethod
    def j_c_H_solver(self, rhs):
        """ solving j_c_h y = rhs
        """


