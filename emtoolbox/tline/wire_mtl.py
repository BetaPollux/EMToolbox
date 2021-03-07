#! /usr/bin/python3

'''Multi-conductor transmission line with round wire conductors'''

import numpy as np
from emtoolbox.utils.constants import MU0, EPS0
import emtoolbox.fields.poisson_fdm as fdm


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
    
    def capacitance(self, /, method: str = None) -> np.ndarray:
        '''Calculate the capacitance matrix'''
        if len(self.conductors) != 2:
            raise Exception('Only two conductors are currently supported')
        w1, w2 = self.conductors
        x1, y1, r1 = w1
        x2, y2, r2 = w2
        s = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if method is None or method.lower() == 'ana':
            return np.array([wire_capacitance(s, r1, r2, er=self.er)])
        elif method.lower() == 'fdm':
            # Define simulation region
            pad = 20  # Zero potential boundaries must be far away
            w_list = np.array([[xi - pad * ri for xi, _, ri in self.conductors],
                               [xi + pad * ri for xi, _, ri in self.conductors]])
            h_list = np.array([[yi - pad * ri for _, yi, ri in self.conductors],
                               [yi + pad * ri for _, yi, ri in self.conductors]])
            # Grid based on wire size or wire separation
            dx = min(min(r1, r2) / 4, s / 12)
            # TODO this grid is unlikely to line up with important edges
            x = np.arange(w_list.min(), w_list.max(), dx)
            y = np.arange(h_list.min(), h_list.max(), dx)
            X, Y = np.meshgrid(x, y)
            print(f'{x.max() - x.min():.3e}, {y.max() - y.min():.3e} dimensions')
            print(f'{X.shape}, {dx:.3e} grid')
            print(f'{X.size} points')
            # Solve for potential
            V1 = 100.0
            bc1 = np.sqrt((X-x1)**2 + (Y-y1)**2) < r1
            bc2 = np.sqrt((X-x2)**2 + (Y-y2)**2) < r2
            bc_val = np.zeros_like(X)
            bc_val[bc1] = 0.5 * V1
            bc_val[bc2] = -0.5 * V1
            bc_bool = bc_val != 0.0
            V = fdm.poisson_2d(X, Y, bc=(bc_bool, bc_val), conv=1e-5)
            er = self.er * np.ones_like(X)[:-1, :-1]
            # Calculate charge and capacitances
            C = np.zeros(len(self.conductors))
            for i, (xi, yi, ri) in enumerate(self.conductors):
                # Keep some distance from wire surface, but cannot overlap other wire
                gpad = min(5 * ri, 0.8 * s)
                xi1 = np.searchsorted(x, xi - gpad)
                xi2 = np.searchsorted(x, xi + gpad)
                yi1 = np.searchsorted(y, yi - gpad)
                yi2 = np.searchsorted(y, yi + gpad)
                print(f'({x[xi1]:.3e}, {x[xi2]:.3e}), ({y[yi1]:.3e}, {y[yi2]:.3e}) gauss')
                C[i] = abs(fdm.gauss_2d(X, Y, V, er, xi1, xi2, yi1, yi2) / V1)
            if abs(C[1] - C[0]) > 1e-12:
                raise Exception(f'Inconsistent capacitance calculation: {C}')
            return np.array([C[0]])
        else:
            raise Exception(f'Invalid method specified: {method}')

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
