#!/usr/bin/python3

import numpy as np
from emtoolbox.utils.constants import EPS0


class SphereCapacitor:
    '''Spherical capacitor with an arbitrary number of dielectric layers'''
    def __init__(self, ri: float, er: float, t: float):
        '''Define geometry as:
            Inner radius            ri : meters
            Dielelctric constants   er : (er1, er2, ...) unitless
            Layer thicknesses       t :  (t1, t2, ...) in meters
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
            self.shells = list(zip(er, t, x0))
        else:
            self.shells = list(zip([er], [t], [ri]))
        self.ri = ri
        self.ro = ri + np.sum(t)

    def capacitance(self):
        '''Get total capacitance in F'''
        C = np.array([EPS0 * 4 * np.pi * er / (1/x0 - 1/(x0 + t))
                      for er, t, x0 in self.shells])
        return 1 / np.sum(1 / C)

    def get_arrays(self, N: int = 11):
        '''Get arrays of position and relative permittivity (X, er)
        X has length N
        er is defined at half-grid points, so has length N-1'''
        X = np.linspace(self.ri, self.ro, N)
        cond = [(X < x0 + t) for _, t, x0 in self.shells]
        choice = [er for er, *_ in self.shells]
        return (X, np.select(cond, choice)[:-1])

    def efield(self, X, Y=None, Z=None, /, Va=1):
        '''Get electric field for the specified differential voltage and locations'''
        if Y is None and Z is None:
            R = X
        else:
            R = np.sqrt(X**2 + Y**2 + Z**2)
        condlist = [R < self.ri, R > self.ro]
        choicelist = [0, 0]
        D = self.charge(Va) / (4 * np.pi * R**2)
        for er, t, x0 in self.shells:
            condlist.insert(-1, R <= x0 + t)
            choicelist.insert(-1, D / (er * EPS0))
        return np.select(condlist, choicelist)

    def potential(self, X, Y=None, Z=None, /, Va=1, Vref=0):
        '''Get potential for the specified differential voltage and locations.
        The reference voltage can be used to offset the results'''
        if Y is None and Z is None:
            R = X
        else:
            R = np.sqrt(X**2 + Y**2 + Z**2)
        R[R == 0.0] = 1e-12
        condlist = [R < self.ri, R > self.ro]
        choicelist = [Vref + Va, Vref]
        V0 = Vref + Va
        Q = self.charge(Va)
        for er, t, x0 in self.shells:
            condlist.insert(-1, R <= x0 + t)
            choicelist.insert(-1, V0 - Q / (4 * np.pi * er * EPS0) * (1/x0 - 1/R))
            V0 -= Q / (4 * np.pi * er * EPS0) * (1/x0 - 1/(x0 + t))
        return np.select(condlist, choicelist)

    def charge(self, Va):
        return self.capacitance() * Va

    def energy(self, Va):
        return 0.5 * self.capacitance() * Va ** 2
