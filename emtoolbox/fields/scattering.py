#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from scipy.special import jv, jvp, hankel2

N0 = 40


def cyl_wave(R, phi, wavelength, E0=1.0, N=N0):
    '''Cylindrical wave polarized in Z-direction'''
    Ez = 0
    k = 2 * np.pi / wavelength
    for n in range(-N, N):
        Ez += E0 * (1.j)**-n * jv(n, k*R) * np.exp(1.j*n*phi)
    return Ez


def scatter_conducting_cylinder(R, phi, wavelength, radius, E0=1.0, N=N0):
    '''Scattered field from a conducting cylinder'''
    Ez = 0
    k = 2 * np.pi / wavelength
    for n in range(-N, N):
        an = -(1.j)**-n * jv(n, k*radius) / hankel2(n, k*radius)
        Ez += E0 * an * hankel2(n, k*R) * np.exp(1.j*n*phi)
    Ez[R < radius] = 0
    return Ez


def scatter_dielectric_cylinder(R, phi, wavelength, radius, er=1.0, ur=1.0, E0=1.0, N=N0):
    '''Scattered field from a dielectric cylinder'''
    Ezs = 0
    Ezint = 0
    k = 2 * np.pi / wavelength
    kd = k * np.sqrt(ur * er)
    a = radius
    for n in range(-N, N):
        hn2 = hankel2(n, k*a)
        dr = 0.001 * k*a
        hn2p = (hankel2(n, k*a + dr) - hankel2(n, k*a - dr)) / (2 * dr)

        an_n = np.sqrt(ur) * jvp(n, k*a) * jv(n, kd*a) - np.sqrt(er) * jv(n, k*a) * jvp(n, kd*a)
        an_d = np.sqrt(ur) * hn2p * jv(n, kd*a) - np.sqrt(er) * hn2 * jvp(n, kd * a)
        an = -(1.j)**-n * an_n / an_d
        Ezs += E0 * an * hankel2(n, k*R) * np.exp(1.j*n*phi)

        cn_n = 2 * np.sqrt(ur)
        cn_d = np.sqrt(ur) * hn2p * jv(n, kd*a) - np.sqrt(er) * hn2 * jvp(n, kd*a)
        cn = (1.j)**-(n+1) / (np.pi * k * a) * cn_n / cn_d
        Ezint += E0 * cn * jv(n, kd*R) * np.exp(1.j*n*phi)
    Ezs[R < a] = Ezint[R < a]
    return Ezint, Ezs


def plot_cyl_wave(ax, X, Y, Ez, title=None, norm=None):
    p = ax.pcolor(X, Y, np.real(Ez), shading='auto', norm=norm)
    ax.set_title(title)
    ax.set_aspect(1)
    ax.get_figure().colorbar(p, ax=ax, orientation='horizontal')


if __name__ == '__main__':
    wavelength = 1.0
    er = 4.0
    r_cyl = wavelength
    w = 4 * wavelength
    x = np.linspace(-w, w, 201)
    y = np.linspace(-w, w, 201)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    Ezi = cyl_wave(R, phi, wavelength)

    Ezs = scatter_conducting_cylinder(R, phi, wavelength, r_cyl)
    Et = Ezi + Ezs
    Et[R < r_cyl] = 0

    Ezint, Ezsd = scatter_dielectric_cylinder(R, phi, wavelength, r_cyl, er=er)
    Etd = Ezi + Ezsd
    Etd[R < r_cyl] = Ezint[R < r_cyl]

    fig, axs = plt.subplots(ncols=3)
    fig.suptitle('Scattering by Conducting Cylinder')
    norm = mpl.colors.Normalize(vmin=-2, vmax=2)
    plot_cyl_wave(axs[0], X, Y, Ezi, 'Incident', norm=norm)
    plot_cyl_wave(axs[1], X, Y, Ezs, 'Scattered', norm=norm)
    plot_cyl_wave(axs[2], X, Y, Et, 'Total', norm=norm)

    fig, axs = plt.subplots(ncols=3)
    fig.suptitle(f'Scattering by Dielectric Cylinder\n$\epsilon_r$ = {er}')
    norm = mpl.colors.Normalize(vmin=-2, vmax=2)
    plot_cyl_wave(axs[0], X, Y, Ezi, 'Incident', norm=norm)
    plot_cyl_wave(axs[1], X, Y, Ezsd, 'Scattered', norm=norm)
    plot_cyl_wave(axs[2], X, Y, Etd, 'Total', norm=norm)

    plt.show()
