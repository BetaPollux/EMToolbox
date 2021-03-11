#! /usr/bin/python3

'''Multi-conductor transmission line with round wire conductors'''

import numpy as np
from emtoolbox.utils.constants import MU0, EPS0
import emtoolbox.fields.poisson_fdm as fdm


class WireMtl():
    def __init__(self, conductors: list, er: float = 1.0, ref: str = 'wire'):
        '''Create a wire-type MTL with the given conductors.
        The conductors are provided as position and radius (x0, y0, rw).
        The conductors are immersed in a relative permittivity of er
        ref is the reference conductor, options are:
            wire
            plane'''
        if ref not in ('wire', 'plane'):
            raise Exception('ref must be either wire or plane')
        if ref == 'wire':
            self.N = len(conductors) - 1
        elif ref == 'plane':
            self.N = len(conductors)
        if self.N < 1:
            raise Exception('Requires at least 2 conductors, or a reference plane')
        for w in conductors:
            if type(w) is not tuple or not list:
                raise Exception(f'Bad conductor type: {w}')
            if len(w) != 3:
                raise Exception(f'Bad conductor definition: {w}')
        # TODO validate positions and radii
        self.conductors = conductors
        self.er = er
        self.ref = ref
    
    def capacitance(self, /, method: str = None, fdm_params: dict = {}) -> np.ndarray:
        '''Calculate the capacitance matrix'''
        x0, y0, r0 = self.conductors[0]
        if method is None or method.lower() == 'ana':
            if self.N == 1:
                x1, y1, r1 = self.conductors[1]
                s = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                return np.array([wire_capacitance(s, r1, r0, self.er)])
            else:
                return MU0 * EPS0 * self.er * np.linalg.inv(self.inductance())
        elif method.lower() == 'fdm':
            if self.N != 1:
                raise Exception('FDM currently supports only two conductors')
            # Define simulation region
            pad = 20  # Zero potential boundaries must be far away
            w_list = np.array([[xi - pad * ri for xi, _, ri in self.conductors],
                               [xi + pad * ri for xi, _, ri in self.conductors]])
            h_list = np.array([[yi - pad * ri for _, yi, ri in self.conductors],
                               [yi + pad * ri for _, yi, ri in self.conductors]])
            # Grid based on wire size or user configured
            x1, y1, r1 = self.conductors[1]
            dx = fdm_params.get('dx', min(r0, r1) / 6)
            x = np.arange(w_list.min(), w_list.max(), dx)
            y = np.arange(h_list.min(), h_list.max(), dx)
            X, Y = np.meshgrid(x, y)
            print(f'{x.max() - x.min():.3e}, {y.max() - y.min():.3e} dimensions')
            print(f'{X.shape}, {dx:.3e} grid')
            print(f'{X.size} points')
            # Solve for potential
            V1 = 100.0
            bc1 = np.sqrt((X-x0)**2 + (Y-y0)**2) <= r0
            bc2 = np.sqrt((X-x1)**2 + (Y-y1)**2) <= r1
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
                # TODO make this more robust
                gpad = ri + 4 * dx
                xi1 = np.searchsorted(x, xi - gpad)
                xi2 = np.searchsorted(x, xi + gpad)
                yi1 = np.searchsorted(y, yi - gpad)
                yi2 = np.searchsorted(y, yi + gpad)
                print(f'({x[xi1]:.3e}, {x[xi2]:.3e}), ({y[yi1]:.3e}, {y[yi2]:.3e}) gauss')
                C[i] = abs(fdm.gauss_2d(X, Y, V, er, xi1, xi2, yi1, yi2) / V1)
            if abs((C[1] - C[0]) / C[0]) > 0.05:
                raise Exception(f'Inconsistent capacitance calculation: {C}')
            return np.array([C[0]])
        else:
            raise Exception(f'Invalid method specified: {method}')

    def inductance(self) -> np.ndarray:
        '''Calculate the inductance matrix'''
        x0, y0, r0 = self.conductors[0]
        d0 = [np.sqrt((xi - x0)**2 + (yi - y0)**2) for xi, yi, _ in self.conductors[1:]]
        if self.N == 1:
            *_, r1 = self.conductors[1]
            L = np.array([wire_inductance(d0[0], r1, r0)])
        else:
            L = np.zeros((self.N, self.N))
            if self.ref == 'wire':
                for i, (xi, yi, ri) in enumerate(self.conductors[1:]):
                    L[i, i] = wire_self_inductance(d0[i], ri, r0)
                    for j, (xj, yj, _) in enumerate(self.conductors[1:]):
                        if j != i:
                            dij = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                            L[i, j] = wire_mutual_inductance(d0[i], d0[j], dij, r0)
                            L[j, i] = L[i, j]
            elif self.ref == 'plane':
                for i, (xi, yi, ri) in enumerate(self.conductors):
                    L[i, i] = wire_self_inductance_plane(yi, ri)
                    for j, (xj, yj, _) in enumerate(self.conductors):
                        if j != i:
                            dij = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                            L[i, j] = wire_mutual_inductance_plane(yi, yj, dij)
                            L[j, i] = L[i, j]
        return L


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


def wire_self_inductance(di0: float, rw0: float, rwi: float):
    '''Inductance between wire i and reference wire 0
    Uses widely separated assumption, di0/rw > 4'''
    return MU0 / (2 * np.pi) * np.log(di0 ** 2 / (rw0 * rwi))


def wire_mutual_inductance(di0: float, dj0: float, dij: float, rw0: float):
    '''Inductance between wire i and wire j, with reference wire 0
    Uses widely separated assumption, di0/rw > 4'''
    return MU0 / (2 * np.pi) * np.log(di0 * dj0 / (dij * rw0))


def wire_self_inductance_plane(hi: float, rwi: float):
    '''Inductance between wire i with height hi above reference plane'''
    return MU0 / (2 * np.pi) * np.log(2 * hi / rwi)


def wire_mutual_inductance_plane(hi: float, hj: float, sij: float):
    '''Inductance between wire i and wire j, above a reference plane'''
    return MU0 / (4 * np.pi) * np.log(1 + 4 * hi * hj / sij**2)
