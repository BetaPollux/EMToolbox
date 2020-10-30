#! /usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


eps0 = 8.8541878176e-12
mu0 = 4e-7 * np.pi


class Material:
    def __init__(self, er=1.0, ur=1.0, cond=0.0,
                 thickness=1e-3, label='Material'):
        self.er = er
        self.ur = ur
        self.cond = cond
        self.thickness = thickness
        self.label = label

    def ec(self, w):
        return eps0 * self.er - 1.j * self.cond / w

    def impedance(self, w):
        return np.sqrt(mu0 * self.ur / self.ec(w))

    def wave_number(self, w):
        return w * np.sqrt(mu0 * self.ur * self.ec(w))

    def skin_depth(self, w):
        return np.sqrt(2 / (w * mu0 * self.ur * self.cond))


def db(values):
    return 20 * np.log10(values)


def reflection_loss(w, medium, shield):
    Zo = medium.impedance(w)
    Zs = shield.impedance(w)
    return np.abs(((Zo + Zs) ** 2) / (4 * Zo * Zs))


def absorption_loss(w, shield):
    return np.abs(np.exp(1.j * shield.wave_number(w) *
                         shield.thickness))


def multiple_reflection_loss(w, medium, shield):
    x = medium.impedance(w) / shield.impedance(w)
    return np.abs(1 - ((x - 1) ** 2 / (x + 1) ** 2) *
                  np.exp(-1.j * 2 * shield.wave_number(w) * shield.thickness))


def shielding_effectiveness(w, medium, shield):
    return (reflection_loss(w, medium, shield) *
            absorption_loss(w, shield) *
            multiple_reflection_loss(w, medium, shield))


def plot_impedance(f, medium, *args):
    w = 2 * np.pi * f
    fig, ax = plt.subplots()
    for mat in args:
        ax.loglog(f, np.abs(mat.impedance(w)), label=mat.label)
    ax.set_title('Impedance')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Impedance (Ohm)')
    ax.legend()
    ax.grid()


def plot_loss(f, medium, *args):
    w = 2 * np.pi * f
    fig, axes = plt.subplots(3, 1, sharex=True)
    for mat in args:
        axes[0].semilogx(f, db(reflection_loss(w, medium, mat)),
                         label=mat.label)
        axes[1].loglog(f, db(absorption_loss(w, mat)),
                       label=mat.label)
        axes[2].semilogx(f, db(multiple_reflection_loss(w, medium, mat)),
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
    w = 2 * np.pi * f
    fig, ax = plt.subplots()
    for mat in args:
        ax.semilogx(f, db(shielding_effectiveness(w, medium, mat)),
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
