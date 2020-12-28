#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def velocity(er):
    return 3e8 / np.sqrt(er)


def beta(w, velocity):
    return w / velocity


def chain_params(char_impedance, beta, length):
    a = np.cos(beta * length)
    b = -1.0j * np.sin(beta * length)
    return np.array([[a, b * char_impedance],
                     [b / char_impedance, a]])


def solve_current(chain, Vs, Zs, Zl):
    n = Zl * chain[1, 0] - chain[0, 0]
    d = chain[0, 1] - chain[0, 0] * Zs - \
        Zl * chain[1, 1] + Zl * chain[1, 0] * Zs
    I0 = n/d * Vs
    Il = chain[1, 0] * Vs + (chain[1, 1] - chain[1, 0] * Zs) * I0
    return np.array([I0, Il])


def solve_voltage(Vs, I0, Il, Zs, Zl):
    V0 = Vs - Zs * I0
    Vl = Zl * Il
    return np.array([V0, Vl])


def create_lowloss(z0, er, length, r=0, g=0):
    v = velocity(er)
    l = z0 / v
    c = 1 / (z0 * v)
    return TLine(l, c, length, r, g)


def create_lossless(z0, er, length):
    return create_lowloss(z0, er, length)


def to_db(x):
    return 20 * np.log10(x)


def freqspace(line, f_min, f_max, max_step):
    # this doesn't work
    steps = []
    f = f_min
    while f < f_max:
        steps.append(f)
        w = 2 * np.pi * f
        f = line.velocity(w) / (line.wavelength(w) * (1 - max_step))
    return np.array(steps)


class TerminatedTLine:
    def __init__(self, tline, zs, zl, vs):
        def make_callable(v):
            if callable(v):
                return v
            else:
                return lambda w: v

        self.tline = tline
        self.zs = make_callable(zs)
        self.zl = make_callable(zl)
        self.vs = make_callable(vs)

    def reflection(self, w, z):
        zc = self.tline.impedance(w)
        refl = (self.zl(w) - zc) / (self.zl(w) + zc)
        return refl * np.exp(2 * self.tline.prop_const(w) *
                             (z - self.tline.length))

    def input_impedance(self, w, z=0):
        refl = self.reflection(w, z)
        return self.tline.impedance(w) * (1 + refl) / (1 - refl)

    def solve(self, w, z):
        zc = self.tline.impedance(w)
        y = self.tline.prop_const(w)
        refl = (self.zl(w) - zc) / (self.zl(w) + zc)
        refs = (self.zs(w) - zc) / (self.zs(w) + zc)
        a = np.exp(-2 * y * self.tline.length)
        b = np.exp(2 * y * z)
        v = (1 + refl * a * b) / (1 - refs * refl * a) * zc
        i = (1 - refl * a * b) / (1 - refs * refl * a)
        return self.vs(w) / (zc + self.zs(w)) * np.exp(-y * z) * np.array([v, i])


class TLine:
    def __init__(self, l, c, length, r=0, g=0):
        self.length = length
        self.r = r
        self.g = g
        self.l = l
        self.c = c

    def wavelength(self, w):
        return 2 * np.pi * self.velocity(w) / w

    def n_wavelengths(self, w):
        return self.length / self.wavelength(w)

    def velocity(self, w):
        return w / self.phase_const(w)

    def delay(self, w):
        return self.length / self.velocity(w)

    def impedance(self, w):
        return np.sqrt((self.r + 1.j * w * self.l) /
                       (self.g + 1.j * w * self.c))

    def prop_const(self, w):
        return np.sqrt((self.r + 1.0j*w*self.l) *
                       (self.g + 1.0j*w*self.c))

    def phase_const(self, w, units=None):
        if units == 'deg' or units == 'deg/m':
            return np.rad2deg(np.imag(self.prop_const(w)))
        return np.imag(self.prop_const(w))

    def attn_const(self, w, units=None):
        if units == 'db' or units == 'db/m':
            return np.real(self.prop_const(w)) * 8.685889638
        return np.real(self.prop_const(w))

    def chain_param(self, w):
        a = np.cosh(self.prop_const(w) * self.length)
        b = np.sinh(self.prop_const(w) * self.length)
        z0 = self.impedance(w)
        return np.array([[a, -b * z0],
                         [-b / z0, a]])


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    f = 100e6
    w = 2 * np.pi * f
    er = 2.0
    wave = velocity(er) / f
    beta = beta(w, velocity(er))
    length = 0.7 * wave
    td = length / velocity(er)
    print('td', td)
    Vs = 1
    Zc = 50
    Zs = 50
    Zl = 100
    cp1 = chain_params(Zc, beta, length)
    cp2 = chain_params(0.5 * Zc, beta, length)
    cp = cp2 @ cp1
    i = solve_current(cp, Vs, Zs, Zl)
    v = solve_voltage(Vs, i[0], i[1], Zs, Zl)
    print('cp1', cp1)
    print('cp2', cp2)
    print('cp', cp)
    print('Beta', beta)
    print('Current', abs(i[0]), abs(i[1]))
    print('Voltage', abs(v[0]), abs(v[1]))

    # line = TLine(4.61e-7, 1.13e-10, 1, 25.46, 1.423e-2)
    # print(line.l, line.c, line.velocity(w), line.delay(w))
    # print(line.attn_const(2*np.pi*1e9))
    # print(line.chain_param(2*np.pi*1e9))
    # print(line.impedance(w))

    print('--- Lossless ---')
    line = create_lossless(Zc, er, length)
    print(line.l, line.c, line.velocity(w), line.delay(w))
    print('attn', line.attn_const(w))
    print('phase', line.phase_const(w))
    print('chain', line.chain_param(w))
    print('impedance', line.impedance(w))

    print('--- Terminated Tline ---')
    network = TerminatedTLine(line, Zs, Zl, Vs)
    src = network.solve(w, 0)
    load = network.solve(w, length)
    print('Zin(0)', abs(network.input_impedance(w)))
    print('Current', abs(src[1]), abs(load[1]))
    print('Voltage', abs(src[0]), abs(load[0]))
    z = np.linspace(0, length, 100)
    fig, ax = plt.subplots()
    ax.plot(z, np.abs(network.solve(w, z)[0, :]), label='V(z)')
    ax.set(xlabel='Position (m)', ylabel='Voltage (V)')
    ax.legend()
    ax.grid()

    n_s = 6
    n_p = 9
    n = 50 * (n_p - n_s)
    f = np.logspace(n_s, n_p, n)
    w = 2 * np.pi * f
    v0 = np.zeros(n)
    vl = np.zeros(n)
    for pt in w:
        cp = line.chain_param(w)
        i = solve_current(cp, Vs, Zs, Zl)
        v = solve_voltage(Vs, i[0], i[1], Zs, Zl)
        v0 = to_db(np.abs(v[0]))
        vl = to_db(np.abs(v[1]))

    fig, ax = plt.subplots()
    ax.semilogx(f, v0, label='V0')
    ax.semilogx(f, vl, label='VL')
    ax.set(xlabel='Frequency (Hz)', ylabel='Voltage (dBV)')
    ax.legend()
    ax.grid()
    plt.show()
