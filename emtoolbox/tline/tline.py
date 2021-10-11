#! /usr/bin/python3

"""
Transmission Line class.

Interal model is based on RLGC parameters at some given frequency.
Parameters can be scalars or an array, when multiple frequencies are provided.


"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from emtoolbox.utils.constants import VP0
except ImportError:
    VP0 = 3e8

DEF_FREQ = 1e6
DEF_ER = 1.0
DEF_LENGTH = 1.0


class TLine:
    def __init__(self, L, C, *, R=0, G=0, freq=DEF_FREQ, length=DEF_LENGTH):
        self.length = length
        self.freq = freq
        self.R = R
        self.G = G
        self.L = L
        self.C = C

    @classmethod
    def create_lowloss(cls, zc, *, freq=DEF_FREQ, **kwargs):
        """Create a lowloss tranmissions line.
        
        zc - characteristic impedance
        freq - frequency for parameters, Hertz

        Options:
        R - resistance, ohms/meter
        G - conductance, siemens/meter
        er - relative dielectric constant, unitless
        vp - velocity of propagation, meters/second
        length - length, meters
        """
        # TODO add support for td, and validate inputs
        if 'er' in kwargs and 'vp' in kwargs:
            raise ValueError('Cannot specify both er and vp')
        elif 'er' in kwargs:
            vp = VP0 / np.sqrt(kwargs['er'])
        elif 'vp' in kwargs:
            vp = kwargs['vp']
        else:
            vp = VP0
        L = zc / vp
        C = 1 / (zc * vp)
        R = kwargs.get('R', 0)
        G = kwargs.get('G', 0)
        length = kwargs.get('length', 1.0)
        return cls(L, C, R=R, G=G, freq=freq, length=length)

    def wavelength(self):
        return self.velocity() / self.freq

    def n_wavelengths(self):
        return self.length / self.wavelength()

    def velocity(self):
        return 2 * np.pi * self.freq / self.phase_const()

    def delay(self):
        return self.length / self.velocity()

    def char_impedance(self):
        return np.sqrt((self.R + 2.j * np.pi * self.freq * self.L) /
                       (self.G + 2.j * np.pi * self.freq * self.C))

    def prop_const(self):
        return np.sqrt((self.R + 2.j * np.pi * self.freq * self.L) *
                       (self.G + 2.j * np.pi * self.freq * self.C))

    def phase_const(self, units=None):
        if units == 'deg' or units == 'deg/m':
            return np.rad2deg(np.imag(self.prop_const()))
        return np.imag(self.prop_const())

    def attn_const(self, units=None):
        if units == 'db' or units == 'db/m':
            return np.real(self.prop_const()) * 8.685889638
        return np.real(self.prop_const())

    def chain_param(self):
        a = np.cosh(self.prop_const() * self.length)
        b = np.sinh(self.prop_const() * self.length)
        zc = self.char_impedance()
        return np.array([[a, -b * zc],
                         [-b / zc, a]])


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    f = 100e6
    er = 2.0
    length = 0.7
    Vs = 1
    Zc = 50
    Zs = 50
    Zl = 100

    print('--- Lossless ---')
    line = TLine.create_lossless(Zc, freq=f, er=er, length=length)
    print(f'{line.L:.3e}, {line.C:.3e}, {line.velocity():.3e}, {line.delay():.3e}')
    print('attn', line.attn_const())
    print('phase', line.phase_const())
    print('chain\n', line.chain_param())
    print('impedance', line.char_impedance())
