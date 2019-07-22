"""Module for constraints class
"""
from __future__ import print_function


class base_constraints(object):
    """ describe constraints for a constrain type equation
    There are 3 Euclidean spaces
    EIn, EOut, EL (lagrange)
    F is a map from EIn to EOut
    C maps from In to EL
    Jacobian_C is an element of EIn^* times EOut
    Jacobian_C^T is an element of EK^* times EIn
    Lagrange multipler is an element of EL
    So we can contract with Jacobian_C^T to give an element in EIn.
    If we have inequality constraints then there is
    ELI for inequality constraints
    Equality is a function of one variable x of type
    tensor (ndarray) and shape shape_in
    returning a tensor of shape shape_out
    """
    _keys = ['J_C', 'J_C2', 'J_C3', 'retraction']
    _state = {}
    
    def __init__(self, shape_in, shape_constraint,
                 equality=None, inequality=None):
        self._shape_in = shape_in
        self._shape_constraint = shape_constraint
        self._equality = equality
        self._inequality = inequality
        self._analytics = {}
        for k in self._keys:
            self._analytics[k] = None

    @property
    def available_states(self):
        return self._state.keys()
    
    @property
    def shape_in(self):
        return self._shape_in

    @property
    def shape_constraint(self):
        return self._shape_constraint

    @property
    def equality(self):
        return self._equality
            
    @property
    def inequality(self):
        return self._inequality

    def set_analytics(self, **kwargs):
        for k in self._keys:
            if k in kwargs:
                self._analytics[k] = kwargs[k]

    def calc_states(self, x):
        """Evaluate C(x) and
        the internal values needed for derivatives
        """
        self._state['C'] = self._equality(x)
        for k in ['J_C', 'J_C2', 'J_C3']:
            if self._analytics[k] is not None:
                self._state[k] = self._analytics[k](x)
        
    def __getitem__(self, key):
        return self._state[key]

    def retraction(self, x, u):
        return self._analytics['retraction'](x, u)
        
    
if __name__ == '__main__':
    c = base_constraints()
    
    
