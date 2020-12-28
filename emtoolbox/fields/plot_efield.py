#!/usr/bin/python3


from electrostatics import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_1d():
    # Plot E-field with radius from a point charge in 1D
    q = 5e-9
    n = 20
    rf_coll = np.linspace(0.4, 10.0, n)
    rq = np.zeros(1)
    E = np.zeros_like(rf_coll)
    for i, rf in enumerate(rf_coll):
        E[i] = efield_point(q, rq, rf)

    fig, ax = plt.subplots()
    ax.plot(rf_coll, E, color='red')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('E-field (V/m)')


def plot_3d_on_axis():
    # Plot E-field from a point charge in 3D along one axis
    q = 5e-9
    n = 20
    rf_coll = np.zeros((n, 3))
    rf_coll[:, 0] = np.linspace(0.4, 10.0, n)
    rq = np.zeros(3)
    E = np.zeros_like(rf_coll)
    for i, rf in enumerate(rf_coll):
        E[i, :] = efield_point(q, rq, rf)

    mag_E = np.linalg.norm(E, axis=1)

    fig, ax = plt.subplots()
    ax.plot(rf_coll[:, 0], mag_E, color='blue')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('E-field (V/m)')


def plot_3d_in_plane():
    # Plot E-field from a point charge within a plane
    q = 5e-9
    n = 100
    x = y = np.linspace(-10.0, 10.0, n)
    X, Y = np.meshgrid(x, y)

    rf_coll = np.zeros((n, n, 3))
    rf_coll[:, :, 0] = X
    rf_coll[:, :, 1] = Y
    rq = np.zeros(3)
    E = np.zeros_like(rf_coll)

    for x in range(n):
        for y in range(n):
            E[x, y, :] = efield_point(q, rq, rf_coll[x, y, :])

    Z = np.linalg.norm(E, axis=2)

    fig, ax = plt.subplots()
    p = ax.pcolormesh(X, Y, Z,
                      norm=mpl.colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                      cmap=mpl.cm.bwr)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('E-field (V/m)')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')


def plot_dipole_in_plane():
    # Plot E-field from a dipole within a plane
    q = [5e-9, -5e-9]
    n = 100
    x = y = np.linspace(-10.0, 10.0, n)
    X, Y = np.meshgrid(x, y)

    rf_coll = np.zeros((n, n, 3))
    rf_coll[:, :, 0] = X
    rf_coll[:, :, 1] = Y
    rq = np.array([[0.0, 1.0, 0.0],
                   [0.0, -1.0, 0.0]])
    E = np.zeros_like(rf_coll)

    for x in range(n):
        for y in range(n):
            E[x, y, :] = efield_point_coll(q, rq, rf_coll[x, y, :])

    Z = np.linalg.norm(E, axis=2)

    fig, ax = plt.subplots()
    p = ax.pcolormesh(X, Y, Z,
                      norm=mpl.colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                      cmap=mpl.cm.bwr)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('E-field (V/m)')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')


if __name__ == '__main__':
    plot_1d()
    plot_3d_on_axis()
    plot_3d_in_plane()
    plot_dipole_in_plane()
    plt.show()
