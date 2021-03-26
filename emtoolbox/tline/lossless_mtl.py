#!/usr/bin/python3

import numpy as np


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
        return 1.0 / np.sqrt(self.L * self.C)

    def delay(self):
        return float(self.length * np.sqrt(self.L * self.C))

    def char_impedance(self):
        return np.sqrt(self.L / self.C)

    def prop_const(self, f):
        return 1.j * self.phase_const(f)

    def phase_const(self, f, units=None):
        beta = (2 * np.pi * f) * float(np.sqrt(self.L * self.C))
        if units == 'deg' or units == 'deg/m':
            return np.rad2deg(beta)
        return beta

    def chain_param(self, f):
        a = np.cos(self.phase_const(f) * self.length)
        b = np.sin(self.phase_const(f) * self.length)
        zc = float(self.char_impedance())
        return np.array([[a,                -1.j * b * zc],
                         [-1.j * b / zc,    a]])
