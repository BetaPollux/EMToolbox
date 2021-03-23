#! /usr/bin/python3

'''Multi-conductor transmission line with round wires'''

import numpy as np
from emtoolbox.utils.constants import MU0, EPS0
from emtoolbox.tline.wire import Wire, Plane, Shield
import emtoolbox.fields.poisson_fdm as fdm


class WireMtl():
    def __init__(self, wires: list, ref, er: float = 1.0):
        '''Create a wire-type MTL with the given wires.
        The wires are provided as Wire objects
        The wires are immersed in a relative permittivity of er
        ref is the reference conductor, of type Wire, Plane or Shield'''
        if type(ref) not in (Wire, Plane, Shield):
            raise TypeError('ref must be wire, plane or shield')
        if len(wires) < 1:
            raise ValueError('Requires at least 2 wires, or a reference plane or shield')
        for w in wires:
            if type(w) is not Wire:
                raise TypeError(f'Bad wire type')
        # TODO validate positions and radii
        self.wires = wires
        self.er = er
        self.ref = ref
    
    def capacitance(self, /, method: str = None, fdm_params: dict = {}) -> np.ndarray:
        '''Calculate the capacitance matrix'''
        if method is None or method.lower() == 'ana':
            return MU0 * EPS0 * self.er * np.linalg.inv(self.inductance())
        elif method.lower() == 'fdm':
            if type(self.ref) is not Wire or len(self.wires) != 1:
                raise Exception('FDM currently supports only two wires, no reference')
            # Define simulation region
            pad = 20  # Zero potential boundaries must be far away
            w_list = np.array([[wi.x - pad * wi.radius for wi in self.wires],
                               [wi.x + pad * wi.radius for wi in self.wires]])
            h_list = np.array([[wi.y - pad * wi.radius for wi in self.wires],
                               [wi.y + pad * wi.radius for wi in self.wires]])
            # Grid based on wire size or user configured
            x0, y0, r0 = self.ref.x, self.ref.y, self.ref.radius
            x1, y1, r1 = self.wires[0].x, self.wires[0].y, self.wires[0].radius
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
            C = np.zeros((len(self.wires), len(self.wires)))
            for i, wi in enumerate(self.wires):
                # Keep some distance from wire surface, but cannot overlap other wire
                # TODO make this more robust
                gpad = wi.radius + 4 * dx
                xi1 = np.searchsorted(x, wi.x - gpad)
                xi2 = np.searchsorted(x, wi.x + gpad)
                yi1 = np.searchsorted(y, wi.y - gpad)
                yi2 = np.searchsorted(y, wi.y + gpad)
                print(f'({x[xi1]:.3e}, {x[xi2]:.3e}), ({y[yi1]:.3e}, {y[yi2]:.3e}) gauss')
                C[i] = abs(fdm.gauss_2d(X, Y, V, er, xi1, xi2, yi1, yi2) / V1)
            return np.array([C[0]])
        else:
            raise Exception(f'Invalid method specified: {method}')

    def inductance(self) -> np.ndarray:
        '''Calculate the inductance matrix'''
        L = np.zeros((len(self.wires), len(self.wires)))
        for i, wi in enumerate(self.wires):
            L[i, i] = wire_self_inductance(wi, self.ref)
            for j, wj in enumerate(self.wires):
                if j != i:
                    L[i, j] = wire_mutual_inductance(wi, wj, self.ref)
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


def wire_self_inductance(wire, ref):
    '''Inductance between wire and reference conductor
    Uses widely separated assumption'''
    if type(ref) == Wire:
        di0 = wire.distance_to(ref)
        rw0 = ref.radius
        rwi = wire.radius
        return MU0 / (2 * np.pi) * np.log(di0 ** 2 / (rw0 * rwi))
    elif type(ref) == Plane:
        hi = ref.height_of(wire)
        rwi = wire.radius
        return MU0 / (2 * np.pi) * np.log(2 * hi / rwi)
    elif type(ref) == Shield:
        rs = ref.radius
        di = wire.offset()
        rwi = wire.radius
        return MU0 / (2 * np.pi) * np.log((rs**2 - di**2) / (rs * rwi))
    else:
        raise Exception('Unrecognized reference type')


def wire_mutual_inductance(wire_i, wire_j, ref):
    # di0: float, dj0: float, dij: float, rw0: float):
    '''Inductance between wire i and wire j, with reference wire 0
    Uses widely separated assumption, di0/rw > 4'''
    if type(ref) == Wire:
        rw0 = ref.radius
        di0 = wire_i.distance_to(ref)
        dj0 = wire_j.distance_to(ref)
        dij = wire_i.distance_to(wire_j)
        return MU0 / (2 * np.pi) * np.log(di0 * dj0 / (dij * rw0))
    elif type(ref) == Plane:
        hi = ref.height_of(wire_i)
        hj = ref.height_of(wire_j)
        sij = wire_i.distance_to(wire_j)
        return MU0 / (4 * np.pi) * np.log(1 + 4 * hi * hj / sij**2)
    elif type(ref) == Shield:
        rs = ref.radius
        di = wire_i.offset()
        dj = wire_j.offset()
        tij = wire_i.angle_to(wire_j)
        return MU0 / (2 * np.pi) * np.log(dj / rs * np.sqrt(
            ((di*dj)**2 + rs**4 - 2* di * dj * rs**2 * np.cos(tij)) /
            ((di*dj)**2 + dj**4 - 2* di * dj**3 * np.cos(tij))
        ))
    else:
        raise Exception('Unrecognized reference type')
