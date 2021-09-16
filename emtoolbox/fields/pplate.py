#!/usr/bin/python3

import numpy as np
from emtoolbox.utils.constants import EPS0


class ParallelPlates:
    '''Parallel Plate one-dimensional structure with an arbitrary number of dielectric layers'''
    def __init__(self, er: float, t: float, area: float = 1):
        '''Define geometry as:
            Dielelctric constants   er : (er1, er2, ...) unitless
            Layer thicknesses       t :  (t1, t2, ...) in meters
            Area                    area: square meters
        '''
        if type(er) != type(t):
            raise Exception('Parameters er and t must be the same type')

        if hasattr(er, '__iter__') and hasattr(t, '__iter__'):
            if len(er) != len(t):
                raise Exception('Parameters er and t must be the same length')
            x0 = []
            next_x0 = 0
            for ti in t:
                x0.append(next_x0)
                next_x0 += ti
            self.slabs = list(zip(er, t, x0))
        else:
            self.slabs = list(zip([er], [t], [0]))

        self.thickness = np.sum(t)
        self.area = area

    def capacitance(self):
        '''Get total capacitance in F'''
        C = np.array([EPS0 * er / t for er, t, _ in self.slabs])
        return self.area / np.sum(1 / C)

    def get_arrays(self, N: int = 11):
        '''Get arrays of position and relative permittivity (X, er)
        X has length N
        er is defined at half-grid points, so has length N-1'''
        X = np.linspace(0, self.thickness, N)
        cond = [(X < x0 + t) for _, t, x0 in self.slabs]
        choice = [er for er, *_ in self.slabs]
        return (X, np.select(cond, choice)[:-1])

    def efield(self, X, Va):
        '''Get electric field for the specified differential voltage and locations'''
        if X.min() < 0 or X.max() > self.thickness:
            raise Exception('Invalid range of X provided')
        D = self.charge(Va) / self.area
        cond = [(X <= x0 + t) for _, t, x0 in self.slabs]
        choice = [-D / (EPS0 * er) for er, *_ in self.slabs]
        return np.select(cond, choice)

    def potential(self, X, Va, Vref=0):
        '''Get potential for the specified differential voltage and locations.
        The reference voltage can be used to offset the results'''
        if X.min() < 0 or X.max() > self.thickness:
            raise Exception('Invalid range of X provided')
        D = self.charge(Va) / self.area
        cond = []
        choice = []
        V0 = Vref
        for er, t, x0 in self.slabs:
            E = D / (EPS0 * er)
            cond.append(X <= x0 + t)
            choice.append(E * (X - x0) + V0)
            V0 += E * t
        return np.select(cond, choice)

    def charge(self, Va):
        return self.capacitance() * Va

    def energy(self, Va):
        return 0.5 * self.capacitance() * Va ** 2
