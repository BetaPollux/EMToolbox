#!/usr/bin/python3

'''Poisson's finite difference method'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def poisson_1d(X, v_left: float=0, v_right: float=0,
               conv: float= 1e-3, Nmax: int=1e5, charge=None):
    '''One-dimension Poisson equation with fixed potential boundaries.
    Normalized charge density ps/eps can be provided via the charge argument'''
    V = np.zeros_like(X)
    V[0] = v_left
    V[-1] = v_right
    V[1:-1] = 0.5 * (v_left + v_right)
    for n in range(int(Nmax)):
        V_old = np.copy(V)
        V[1:-1] = 0.5 * (V[2:] + V[:-2])
        if charge is not None:
            V -= 0.5 * charge
        err = np.sum(np.abs(V - V_old))
        if err < conv:
            break
    print(f'1D Error {err:.3e} after {n+1} iterations')
    return V


def poisson_2d(X, Y,
               v_left: float=0, v_right: float=0,
               v_top: float=0, v_bottom: float=0,
               conv: float= 1e-3, Nmax: int=1e5, charge=None):
    '''Two-dimension Poisson equation with fixed potential boundaries.
    Normalized charge density ps/eps can be provided via the charge argument'''
    V = np.zeros_like(X)
    V[:, 0] = v_left
    V[:, -1] = v_right
    V[-1, :] = v_top
    V[0, :] = v_bottom
    V[0, 0] = 0.5 * (v_bottom + v_left)
    V[0, -1] = 0.5 * (v_bottom + v_right)
    V[-1, 0] = 0.5 * (v_top + v_left)
    V[-1, -1] = 0.5 * (v_top + v_right)
    V[1:-1, 1:-1] = 0.25 * (v_bottom + v_right + v_top + v_left)
    for n in range(int(Nmax)):
        V_old = np.copy(V)
        V[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] +
                                V[1:-1, 2:] + V[1:-1, :-2])
        if charge is not None:
            V -= 0.25 * charge
        err = np.sum(np.abs(V - V_old))
        if err < conv:
            break
    print(f'2D Error {err:.3e} after {n+1} iterations')
    return V


def plates_analytical(X, v_left: float=0, v_right: float=0):
    return (v_right - v_left) / np.max(X) * X + v_left


def trough_analytical(X, Y,
                      v_left: float=0, v_right: float=0,
                      v_top: float=0, v_bottom: float=0):
    a = np.max(X)
    b = np.max(Y)
    V = np.zeros_like(X)
    for n in range(1, 101, 2):
        k1 = 4 / (n * np.pi) * np.sin(n * np.pi * Y / b) / np.sinh(n * np.pi * a / b)
        k2 = 4 / (n * np.pi) * np.sin(n * np.pi * X / a) / np.sinh(n * np.pi * b / a)
        vx = v_right * np.sinh(n * np.pi * X / b) + v_left * np.sinh(n * np.pi / b * (a - X))
        vy = v_top * np.sinh(n * np.pi * Y / a) + v_bottom * np.sinh(n * np.pi / a * (b - Y))
        V += k1 * vx + k2 * vy
    return V


def example_poisson_1d():
    def dist(x, a0, m, std):
        return a0 * np.exp(-0.5 * ((x - m) / std) ** 2)

    w = 1.0
    x = np.linspace(0, w, 100)
    v_left = -2
    v_right = 1
    qp = dist(x, 0.01, 0.25 * w, 0.02 * w)
    qn = dist(x, -0.02, 0.75 * w, 0.05 * w)
    Q = (None, qp, qp + qn)

    fig, axes = plt.subplots(len(Q), 1, sharex=True)
    fig.suptitle(f'Finite Difference Method (1D)\n')
    for i, Qi in enumerate(Q):
        V = poisson_1d(x, v_left, v_right, charge=Qi)
        axes[i].plot(x, V, color='b')
        if Qi is not None:
            q_axis = axes[i].twinx()
            q_axis.plot(x, Qi, lw=0.5, color='r')
            q_axis.set_ylabel(r'$\rho$ / $\epsilon$', color='red')
        axes[i].set_ylabel('Potential (V)', color='b')
        axes[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    axes[-1].set_xlim([0, w])
    plt.show()


def example_poisson_2d():
    w = 2.0
    h = 1.0
    x = np.linspace(0, w, 101)
    y = np.linspace(0, h, 51)
    X, Y = np.meshgrid(x, y)
    bc = {'v_top': 10, 'v_left': 5}
    V = poisson_2d(X, Y, **bc)
    Va = trough_analytical(X, Y, **bc)
    error = np.abs(Va - V) + 1e-10

    fig = plt.figure()
    fig.suptitle('Poisson Equation 2D')
    grid_spec = plt.GridSpec(2, 2, hspace=0.4)
    ax_fdm = fig.add_subplot(grid_spec[0])
    ax_ana = fig.add_subplot(grid_spec[1])
    ax_err = fig.add_subplot(grid_spec[2])
    ax_surf = fig.add_subplot(grid_spec[3], projection='3d')
    ax_fdm.set_title('Finite Difference')
    ax_fdm.contour(X, Y, V)
    ax_ana.set_title('Analytical Solution')
    ax_ana.contour(X, Y, Va)
    ax_err.set_title('|Error|')
    c = ax_err.pcolor(X, Y, error, shading='auto',
                   norm=mpl.colors.LogNorm(vmin=error.min(), vmax=error.max()), cmap='PuBu_r')
    fig.colorbar(c, ax=ax_err)
    ax_surf.set_title('Surface')
    ax_surf.plot_surface(X, Y, V)
    plt.show()


if __name__ == '__main__':
    example_poisson_1d()
    example_poisson_2d()
