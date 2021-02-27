#!/usr/bin/python3

'''Poisson's finite difference method'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    from emtoolbox.utils.constants import EPS0
except ImportError:
    EPS0 = 8.854e-12


def poisson_1d(X: np.ndarray, /, v_left: float=0, v_right: float=0,
               dielectric: np.ndarray=None, charge: np.ndarray=None,
               bc: list=None,
               conv: float= 1e-3, Nmax: int=1e5):
    '''One-dimension Poisson equation with fixed potential boundaries.
    Normalized charge density ps/eps can be provided via the charge argument
    Dielectric is an array of relative permittivity, located at half-grid points
        x0    x1    x2  ...  xn
           e0    e1  ...  en-1   [one less point]
    Note: Charge density is currently incompatible with dielectric'''
    if charge is not None and dielectric is not None:
        raise Exception('Charge is not support with dielectric')
    V = np.zeros_like(X)
    V[0] = v_left
    V[-1] = v_right
    V[1:-1] = 0.5 * (v_left + v_right)
    for n in range(int(Nmax)):
        V_old = np.copy(V)
        if dielectric is None:
            V[1:-1] = 0.5 * (V[2:] + V[:-2])
            if charge is not None:
                V[1:-1] -= 0.5 * charge[1:-1]
        else:
            er1 = dielectric[:-1]
            er2 = dielectric[1:]
            V[1:-1] = (er2 * V[2:] + er1 * V[:-2]) / (er1 + er2)
        if bc:
            for bx, bv in bc:
                V[bx] = bv
        err = np.sum(np.abs(V - V_old))
        if err < conv:
            break
    print(f'1D Error {err:.3e} after {n+1} iterations')
    return V


def poisson_2d(X: np.ndarray, Y: np.ndarray, /,
               v_left: float=0, v_right: float=0,
               v_top: float=0, v_bottom: float=0,
               dielectric: np.ndarray=None, charge: np.ndarray=None,
               bc: list=None,
               conv: float= 1e-3, Nmax: int=1e5):
    '''Two-dimension Poisson equation with fixed potential boundaries.
    Normalized charge density ps/eps can be provided via the charge argument'''
    if X[0, 1] == X[0, 0] or Y[1, 0] == Y[0, 0]:
        raise Exception('X and Y must have xy indexing')
    if X[0, 1] - X[0, 0] != Y[1, 0] - Y[0, 0]:
        raise Exception('X and Y must have the same spacing')
    if charge is not None and dielectric is not None:
        raise Exception('Charge is not support with dielectric')
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
        if dielectric is None:
            V[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] +
                                    V[1:-1, 2:] + V[1:-1, :-2])
            if charge is not None:
                V[1:-1, 1:-1] -= 0.25 * charge[1:-1, 1:-1]
        else:
            er_nw = dielectric[1:, :-1]
            er_ne = dielectric[1:, 1:]
            er_sw = dielectric[:-1, :-1]
            er_se = dielectric[:-1:, 1:]
            V[1:-1, 1:-1] = (((er_sw + er_nw) * V[1:-1, :-2] +
                              (er_nw + er_ne) * V[2:, 1:-1] +
                              (er_ne + er_se) * V[1:-1, 2:] +
                              (er_se + er_sw) * V[:-2, 1:-1]) /
                              (2 * (er_nw + er_ne + er_sw + er_se)))
        if bc:
            for bxy, bv in bc:
                V[bxy] = bv
        err = np.sum(np.abs(V - V_old))
        if err < conv:
            break
    print(f'2D Error {err:.3e} after {n+1} iterations')
    return V


def gauss_1d(X: np.ndarray, V: np.ndarray, er: np.ndarray, i: int):
    '''One-dimensional Gauss' law, returning enclosed charge
    Evaluated at array index i
    Note: charge polarity is positive for V increasing with X'''
    dx = X[i+1] - X[i-1]
    return EPS0 * (er[i] * V[i+1] - er[i-1] * V[i-1] + 
                   (er[i-1] - er[i]) * V[i]) / dx


def trough_analytical(X: np.ndarray, Y: np.ndarray,
                      v_left: float=0, v_right: float=0,
                      v_top: float=0, v_bottom: float=0):
    a = X.max() - X.min()
    b = Y.max() - Y.min()
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


def example_parallel_plates():
    w = 4e-3
    X, dx = np.linspace(0, w, 101, retstep=True)
    bc = {'v_left': 0, 'v_right': 200}
    er1 = 5.0
    er2 = 1.0
    era = np.where(X[:-1] < 1e-3, er1, er2)
    erb = np.select([X[:-1] < 2e-3, X[:-1] < 3e-3, X[:-1] < 4e-3],
                    [er2, er1, er2])

    V1 = poisson_1d(X, dielectric=era, **bc)
    V2 = poisson_1d(X, dielectric=erb, **bc)

    _, ax = plt.subplots()
    ax.plot(X, V1, label='2-layer')
    ax.plot(X, V2, label='3-layer')
    ax.set_ylabel('Potential (V)')
    ax.set_xlim([0, w])
    er_ax = ax.twinx()
    er_ax.plot(X[:-1] + 0.5 * dx, era, ls=':')
    er_ax.plot(X[:-1] + 0.5 * dx, erb, ls=':')
    er_ax.set_ylabel(r'$\epsilon_r$')
    ax.legend()
    ax.grid()
    plt.show()


def example_poisson_1d_bc():
    X = np.linspace(0, 10, 21)
    v0 = -2
    v1 = 3
    v2 = 6
    bc = (([2, 3, 4], v1),)
    V = poisson_1d(X, v_left=v0, v_right=v2, bc=bc, conv=1e-3)
    plt.plot(X, V)
    plt.grid()
    plt.show()


def example_poisson_2d_coax():
    ri = 1.5e-3
    ro = 4.0e-3
    w = 1.1 * ro
    N = 101
    Va = 10.0
    x = np.linspace(-w, w, N)
    y = np.linspace(-w, w, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    bc = ((R <= ri, Va), (R >= ro, 0))
    V = poisson_2d(X, Y, bc=bc)
    _, ax = plt.subplots()
    ax.contour(X, Y, V)
    ax.set_aspect('equal')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    example_poisson_1d()
    example_poisson_2d()
    example_parallel_plates()
    example_poisson_1d_bc()
    example_poisson_2d_coax()
