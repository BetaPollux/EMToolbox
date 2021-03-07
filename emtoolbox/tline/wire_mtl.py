#! /usr/bin/python3

'''Multi-conductor transmission line with round wire conductors'''

import numpy as np
from emtoolbox.utils.constants import MU0, EPS0


class WireMtl():
    def __init__(self, conductors: list, er: float = 1.0):
        '''Create a wire-type MTL with the given conductors.
        The conductors are provided as position and radius (x0, y0, rw).
        The conductors are immersed in a relative permittivity of er'''
        if len(conductors) < 2:
            raise Exception('Requires at least 2 conductors')
        for w in conductors:
            if type(w) is not tuple or not list:
                raise Exception(f'Bad conductor type: {w}')
            if len(w) != 3:
                raise Exception(f'Bad conductor definition: {w}')
        self.conductors = conductors
        self.er = er
    
    def capacitance(self) -> np.ndarray:
        '''Calculate the capacitance matrix'''
        if len(self.conductors) != 2:
            raise Exception('Only two conductors are currently supported')
        w1, w2 = self.conductors
        x1, y1, r1 = w1
        x2, y2, r2 = w2
        s = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return np.array([wire_capacitance(s, r1, r2, er=self.er)])
    
    def inductance(self) -> np.ndarray:
        '''Calculate the inductance matrix'''
        if len(self.conductors) != 2:
            raise Exception('Only two conductors are currently supported')
        w1, w2 = self.conductors
        x1, y1, r1 = w1
        x2, y2, r2 = w2
        s = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return np.array([wire_inductance(s, r1, r2)])


def wire_capacitance(s: float, rw1: float, rw2: float = None, er: float = 1.0) -> float:
    '''Capacitance between two wires, immersed in a homogenous dielectric medium
    Center-to-center separation s with individual wire radii rw1 and rw2.
    Units are arbitrary but must be consistent'''
    if rw2:
        if s < rw1 + rw2:
            raise Exception('Separation must be greater than sum of radii')
        return 2 * np.pi * EPS0 * er / (
            np.arccosh((s**2 - rw1**2 - rw2**2) / (2 * rw1 * rw2)))
    else:
        if s < 2 * rw1:
            raise Exception('Separation must be greater than sum of radii')
        return np.pi * EPS0 * er / np.arccosh(s / (2 * rw1))


def wire_inductance(s: float, rw1: float, rw2: float = None) -> float:
    '''Inductance between two wires in a non-magnetic medium
    Center-to-center separation s with individual wire radii rw1 and rw2.
    Units are arbitrary but must be consistent'''
    if rw2:
        if s < rw1 + rw2:
            raise Exception('Separation must be greater than sum of radii')    
        return MU0 / (2 * np.pi) * (
            np.arccosh((s**2 - rw1**2 - rw2**2) / (2 * rw1 * rw2)))
    else:
        if s < 2 * rw1:
            raise Exception('Separation must be greater than sum of radii')
        return MU0 / np.pi * np.arccosh(s / (2 * rw1))
