from __future__ import print_function
import numpy as np
from scipy.optimize import root
from .constraints import base_constraints


class parametrized_constraints(base_constraints):
    """ supplying a number n_f of functions
    and its derivatives x is a vector
    of size n_var + n_func
    constraints given by x[:n_var+i] = funcs[i](x:[n_var])
    """
    
    def __init__(
            self, shape_in, funcs,
            derivs, second_derivs=None):
        assert(len(funcs) == len(derivs))
        shape_constraint = (len(funcs),)
        super(parametrized_constraints, self).__init__(
            shape_in, shape_constraint)

        self.funcs = funcs
        self.derivs = derivs
        self.second_derivs = second_derivs
        
        self._n_func = shape_constraint[0]
        self._n_var = shape_in[0] - self._n_func
        
    def calc_states(self, x):
        self._state['C'] = self.equality(x)
        self._state['J_C'] = self.calc_J_C(x)
        if self.second_derivs is not None:
            self._state['J_C2'] = self.calc_J_C2(x)

    def equality(self, x):
        ret = np.zeros(self.shape_constraint)
        for i in range(self._n_func):
            ret[i] = self.funcs[i](x[:self._n_var]) - x[self._n_var + i]
        return ret

    def calc_J_C(self, x):
        ret = np.zeros((self.shape_constraint[0],
                        self.shape_in[0]))
        for i in range(self._n_func):
            ret[i, :self._n_var] = self.derivs[i](x[:self._n_var])
            ret[i, self._n_var + i] = -1
        return ret

    def calc_J_C2(self, x):
        ret = np.zeros((self.shape_constraint[0],
                        self.shape_in[0],
                        self.shape_in[0]))
        for i in range(self._n_func):
            ret[i, :self._n_var, :self._n_var] =\
                self.second_derivs[i](x[:self._n_var])
        return ret
        
    def retraction(self, x, h):
        """ Do orthographic retraction
        """
        J_C_X = self.calc_J_C(x)
        
        def a_func(t):
            return self.equality(x + h + np.dot(
                J_C_X.T, t))
            
        def f_prime(t):
            return np.dot(self.calc_J_C(x+h+np.dot(
                J_C_X.T, t)), J_C_X.T)

        t0 = np.full((self._n_func), 0.01)
        res = root(a_func, t0, jac=f_prime)
        # print res
        if res['success'] or (np.linalg.norm(res['fun']) < 1.e-5):
            return x + h + np.dot(J_C_X.T, res['x'])
        else:
            print('Failed')
            print(res)
            return None

        
        
