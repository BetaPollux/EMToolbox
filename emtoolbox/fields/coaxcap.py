#!/bin/usr/python3

import numpy as np
from emtoolbox.utils.constants import EPS0


class CoaxCapacitor:
    '''Coaxial capacitor with an arbitrary number of dielectric layers'''
    def __init__(self, ri: float, er: float, t: float, length: float = 1):
        '''Define geometry as:
            Inner radius            ri : meters
            Dielelctric constants   er : (er1, er2, ...) unitless
            Layer thicknesses       t :  (t1, t2, ...) in meters
            Length                  length: meters
        '''
        if type(er) != type(t):
            raise Exception('Parameters er and t must be the same type')

        if hasattr(er, '__iter__') and hasattr(t, '__iter__'):
            if len(er) != len(t):
                raise Exception('Parameters er and t must be the same length')
            x0 = []
            next_x0 = ri
            for ti in t:
                x0.append(next_x0)
                next_x0 += ti
            self.sheaths = list(zip(er, t, x0))
        else:
            self.sheaths = list(zip([er], [t], [ri]))

        self.ri = ri
        self.ro = ri + np.sum(t)
        self.length = length

    def capacitance(self):
        '''Get total capacitance in F'''
        C = np.array([EPS0 * 2 * np.pi * er / np.log((x0 + t) / x0)
                      for er, t, x0 in self.sheaths])
        return self.length / np.sum(1 / C)

    def get_arrays(self, N: int = 11):
        '''Get arrays of position and relative permittivity (X, er)
        X has length N
        er is defined at half-grid points, so has length N-1'''
        X = np.linspace(self.ri, self.ro, N)
        cond = [(X < x0 + t) for _, t, x0 in self.sheaths]
        choice = [er for er, *_ in self.sheaths]
        return (X, np.select(cond, choice)[:-1])

    def efield(self, X, Va, Y=None):
        '''Get electric field for the specified differential voltage and locations'''
        if X.min() < self.ri or X.max() > self.ro:
            raise Exception('Invalid range of X provided')
        if len(self.sheaths) > 1:
            raise Exception('Multiple dielectrics not yet supported')
        return -Va / X / np.log(self.ri / self.ro)

    def potential(self, X, Y=None, /, Va=1, Vref=0):
        '''Get potential for the specified differential voltage and locations.
        The reference voltage can be used to offset the results'''
        if Y is None:
            R = X
        else:
            R = np.sqrt(X**2 + Y**2)
        if len(self.sheaths) > 1:
            raise Exception('Multiple dielectrics not yet supported')
        condlist = [R < self.ri, R <= self.ro, R > self.ro]
        choicelist = [Vref + Va,
                      Vref + Va / np.log(self.ri / self.ro) * np.log(R / self.ro, where=(R>=self.ri)),
                      Vref]
        return np.select(condlist, choicelist)

    def charge(self, Va):
        return self.capacitance() * Va

    def energy(self, Va):
        return 0.5 * self.capacitance() * Va ** 2
 