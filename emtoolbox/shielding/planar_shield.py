#! /usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from emtoolbox.utils.constants import EPS0, MU0
except ImportError:
    EPS0 = 8.854e-12
    MU0 = 4e-7 * np.pi


class Material:
    def __init__(self, er=1.0, ur=1.0, cond=0.0,
                 thickness=1e-3, label='Material'):
        self.er = er
        self.ur = ur
        self.cond = cond
        self.thickness = thickness
        self.label = label

    def ec(self, f):
        return EPS0 * self.er - 1.j * self.cond / (2 * np.pi * f)

    def impedance(self, f):
        return np.sqrt(MU0 * self.ur / self.ec(f))

    def wave_number(self, f):
        return 2 * np.pi * f * np.sqrt(MU0 * self.ur * self.ec(f))

    def skin_depth(self, f):
        return np.sqrt(1 / (f * np.pi * MU0 * self.ur * self.cond))


def db(values):
    return 20 * np.log10(values)


def reflection_loss(f, medium, shield):
    Zo = medium.impedance(f)
    Zs = shield.impedance(f)
    return np.abs(((Zo + Zs) ** 2) / (4 * Zo * Zs))


def absorption_loss(f, shield):
    return np.abs(np.exp(1.j * shield.wave_number(f) *
                         shield.thickness))


def multiple_reflection_loss(f, medium, shield):
    x = medium.impedance(f) / shield.impedance(f)
    return np.abs(1 - ((x - 1) ** 2 / (x + 1) ** 2) *
                  np.exp(-1.j * 2 * shield.wave_number(f) * shield.thickness))


def shielding_effectiveness(f, medium, shield):
    return (reflection_loss(f, medium, shield) *
            absorption_loss(f, shield) *
            multiple_reflection_loss(f, medium, shield))


def plot_impedance(f, medium, *args):
    fig, ax = plt.subplots()
    for mat in args:
        ax.loglog(f, np.abs(mat.impedance(f)), label=mat.label)
    ax.set_title('Impedance')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Impedance (Ohm)')
    ax.legend()
    ax.grid()


def plot_loss(f, medium, *args):
    fig, axes = plt.subplots(3, 1, sharex=True)
    for mat in args:
        axes[0].semilogx(f, db(reflection_loss(f, medium, mat)),
                         label=mat.label)
        axes[1].loglog(f, db(absorption_loss(f, mat)),
                       label=mat.label)
        axes[2].semilogx(f, db(multiple_reflection_loss(f, medium, mat)),
                         label=mat.label)

    axes[0].set_title('Loss Terms')
    axes[0].set_ylabel('Reflection (dB)')
    axes[0].legend()
    axes[0].grid()

    axes[1].set_ylabel('Absorption (dB)')
    axes[1].legend()
    axes[1].grid(which='both', axis='y', linestyle=':')
    axes[1].grid(which='major', axis='x')

    axes[2].set_ylabel('Multiple Reflection (dB)')
    axes[2].legend()
    axes[2].grid()
    axes[2].set_xlabel('Frequency (Hz)')


def plot_shielding(f, medium, *args):
    fig, ax = plt.subplots()
    for mat in args:
        ax.semilogx(f, db(shielding_effectiveness(f, medium, mat)),
                    label=mat.label)

    ax.set_title('Shielding Effectiveness')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Shielding Effectiveness (dB)')
    ax.legend()
    ax.grid()


if __name__ == '__main__':
    air = Material()
    thk = 0.25e-3
    copper = Material(cond=11.8e6, thickness=thk, label='Copper')
    duranickel = Material(ur=10.58, cond=2.3e6, thickness=thk,
                          label='Duranickel')
    ssteel = Material(ur=95, cond=1.3e6, thickness=thk,
                      label='Stainless Steel')

    f = np.logspace(4, 9, 50)
    plot_impedance(f, air, copper, duranickel, ssteel)
    plot_loss(f, air, copper, duranickel, ssteel)
    plot_shielding(f, air, copper, duranickel, ssteel)

    lossy = Material(er=4.0, cond=0.04, thickness=0.5, label='Lossy')
    plot_loss(f, air, lossy)
    plot_shielding(f, air, lossy)

    plt.show()
