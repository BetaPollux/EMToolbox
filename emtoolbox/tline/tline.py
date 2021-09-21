#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
try:
    from emtoolbox.utils.constants import VP0
except ImportError:
    VP0 = 3e8


class TerminatedTLine:
    def __init__(self, tline, zs, zl, vs):
        self.tline = tline
        self.zs = zs
        self.zl = zl
        self.vs = vs

    def reflection(self, z):
        zc = self.tline.impedance()
        refl = (self.zl - zc) / (self.zl + zc)
        return refl * np.exp(2 * self.tline.prop_const() *
                             (z - self.tline.length))

    def input_impedance(self, z=0):
        refl = self.reflection(z)
        return self.tline.impedance() * (1 + refl) / (1 - refl)

    def solve(self, z):
        zc = self.tline.impedance()
        y = self.tline.prop_const()
        refl = (self.zl - zc) / (self.zl + zc)
        refs = (self.zs - zc) / (self.zs + zc)
        a = np.exp(-2 * y * self.tline.length)
        b = np.exp(2 * y * z)
        v = (1 + refl * a * b) / (1 - refs * refl * a) * zc
        i = (1 - refl * a * b) / (1 - refs * refl * a)
        return self.vs / (zc + self.zs) * np.exp(-y * z) * np.array([v, i])


class TLine:
    def __init__(self, freq, L, C, length=1.0, R=0, G=0):
        self.length = length
        self.freq = freq
        self.R = R
        self.G = G
        self.L = L
        self.C = C

    @classmethod
    def create_lowloss(cls, freq, zc, er, length=1.0, R=0, G=0):
        vp = VP0 / np.sqrt(er)
        L = zc / vp
        C = 1 / (zc * vp)
        return cls(freq, L, C, length=length, R=R, G=G)

    @classmethod
    def create_lossless(cls, freq, zc, er, length=1.0):
        return cls.create_lowloss(freq, zc, er, length)

    def wavelength(self):
        return self.velocity() / self.freq

    def n_wavelengths(self):
        return self.length / self.wavelength()

    def velocity(self):
        return 2 * np.pi * self.freq / self.phase_const()

    def delay(self):
        return self.length / self.velocity()

    def impedance(self):
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
        zc = self.impedance()
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
    line = TLine.create_lossless(f, Zc, er, length)
    print(f'{line.L:.3e}, {line.C:.3e}, {line.velocity():.3e}, {line.delay():.3e}')
    print('attn', line.attn_const())
    print('phase', line.phase_const())
    print('chain\n', line.chain_param())
    print('impedance', line.impedance())

    print('--- Terminated Tline ---')
    network = TerminatedTLine(line, Zs, Zl, Vs)
    z = np.linspace(0, length, 100)
    sol = network.solve(z)
    src = sol[:, 0]
    load = sol[:, -1]
    print(f'Zin(0) {abs(network.input_impedance()):.3f}')
    print(f'Current {abs(src[1]):.3f} {abs(load[1]):.3f}')
    print(f'Voltage {abs(src[0]):.3f} {abs(load[0]):.3f}')
    
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(z, np.abs(sol[0, :]), label='V(z)')
    axs[0].set(xlabel='Position (m)', ylabel='Voltage (V)')
    axs[1].plot(z, 1000 * np.abs(sol[1, :]), label='I(z)')
    axs[1].set(xlabel='Position (m)', ylabel='Current (mA)')
    for ax in axs:
        ax.legend()
        ax.grid()

    plt.show()
