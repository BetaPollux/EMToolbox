#!/usr/bin/python3

import numpy as np
from emtoolbox.utils.constants import VP0


class ZcMtl():
    def __init__(self, zc, velocity=VP0):
        self.velocity = velocity
        self.L = zc / self.velocity
        self.C = 1 / (zc * self.velocity)

    def inductance(self):
        return np.array([[self.L]])

    def capacitance(self):
        return np.array([[self.C]])


class LosslessMtl():
    def __init__(self, mtl, length=1.0):
        self.L = mtl.inductance()
        self.C = mtl.capacitance()
        self.length = length

    def wavelength(self, f):
        return self.velocity() / f

    def n_wavelengths(self, f):
        return self.length / self.wavelength(f)

    def velocity(self):
        return float(1.0 / np.sqrt(self.L * self.C))

    def delay(self):
        return float(self.length * np.sqrt(self.L * self.C))

    def char_impedance(self):
        return float(np.sqrt(self.L / self.C))

    def prop_const(self, f):
        return 1.j * self.phase_const(f)

    def phase_const(self, f, units=None):
        beta = (2 * np.pi * f) * float(np.sqrt(self.L * self.C))
        if units == 'deg' or units == 'deg/m':
            return np.rad2deg(beta)
        return beta

    def attn_const(self, f):
        return np.zeros_like(f)

    def chain_param(self, f):
        a = np.cos(self.phase_const(f) * self.length)
        b = np.sin(self.phase_const(f) * self.length)
        zc = self.char_impedance()
        return np.array([[a,                -1.j * b * zc],
                         [-1.j * b / zc,    a]])
