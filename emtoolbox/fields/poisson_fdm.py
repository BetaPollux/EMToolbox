#!/usr/bin/python3

'''Poisson's finite difference method'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
try:
    from emtoolbox.utils.constants import EPS0
except ImportError:
    EPS0 = 8.854e-12


@jit()
def check_arrays_1d(X, er=None):
    if er is not None:
        if er.shape[0] != X.shape[0] - 1:
            raise Exception('X shape must be one larger than er')


@jit()
def check_arrays_2d(X, Y, er=None):
    if X.shape != Y.shape:
        raise Exception('X and Y shape must identical')
    if X[1, 0] == X[0, 0] or Y[0, 1] == Y[0, 0]:
        raise Exception('X and Y must have ij indexing')
    if abs(abs(X[1, 0] - X[0, 0]) - abs(Y[0, 1] - Y[0, 0])) > 1e-6:
        raise Exception('X and Y must have the same spacing')
    if er is not None:
        if er.shape[0] != X.shape[0] - 1 or er.shape[1] != X.shape[1] - 1:
            raise Exception('X and Y shape must be one larger than er')


@jit()
def check_arrays_3d(X, Y, Z):
    if X[1, 0, 0] == X[0, 0, 0] or Y[0, 1, 0] == Y[0, 0, 0] or Z[0, 0, 1] == Z[0, 0, 0]:
        raise Exception('X, Y and Z must have ij indexing')
    # TODO check array spacing


@jit(nopython=True)
def poisson_1d(X: np.ndarray, /, v_left: float = 0, v_right: float = 0,
               dielectric: np.ndarray = None, charge: np.ndarray = None,
               bc: list = None, sor=1.8,
               conv: float = 1e-5, Nmax: int = 1e5):
    '''One-dimension Poisson equation with fixed potential boundaries.
    Normalized charge density ps/eps can be provided via the charge argument
    Dielectric is an array of relative permittivity, located at half-grid points
        x0    x1    x2  ...  xn
           e0    e1  ...  en-1   [one less point]
    Boundary condition is to be provided as a (bool array, value array) matching X
    Where the condition is X[bool] = value'''
    check_arrays_1d(X, dielectric)
    if charge is not None:
        raise Exception('Charge is currently not supported')
    # TODO enforce array types
    V = np.zeros_like(X, dtype='float64')
    V[0] = v_left
    V[-1] = v_right
    V[1:-1] = 0.5 * (v_left + v_right)  # Initial seed
    nx = len(X)
    if bc is None:
        # Explicit to prompt numba type
        bc_bool = np.array([False])
        bc_val = np.array([0.0])
    else:
        bc_bool, bc_val = bc
        for i in range(nx):
            if bc_bool[i]:
                V[i] = bc_val[i]
    for n in range(int(Nmax)):
        Vsum = 0
        Verr = 0
        for i in range(1, nx-1):
            V_old = V[i]
            if not bc or not bc_bool[i]:
                if dielectric is None:
                    R = 0.5 * (V[i+1] + V[i-1]) - V_old
                else:
                    er1 = dielectric[i-1]
                    er2 = dielectric[i]
                    R = (er2 * V[i+1] + er1 * V[i-1]) / (er1 + er2) - V_old
                V[i] = R * sor + V_old
                Verr += abs(R)
                Vsum += abs(V[i])
        if Vsum > 0 and Verr / Vsum < conv:
            break
    print('1D Error', Verr / Vsum, 'after', n+1, 'iterations')
    return V


@jit(nopython=True)
def poisson_2d(X: np.ndarray, Y: np.ndarray, /,
               v_left: float = 0, v_right: float = 0,
               v_top: float = 0, v_bottom: float = 0,
               dielectric: np.ndarray = None, charge: np.ndarray = None,
               bc: list = None, sor=1.8, xsym: bool = False, ysym: bool = False,
               conv: float = 1e-5, Nmax: int = 1e5):
    '''Two-dimension Poisson equation with fixed potential boundaries.
    Normalized charge density ps/eps can be provided via the charge argument'''
    check_arrays_2d(X, Y, dielectric)
    if charge is not None:
        raise Exception('Charge is currently not supported')
    # TODO enforce array types
    V = np.zeros_like(X, dtype='float64')
    V[0, :] = v_left
    V[-1, :] = v_right
    V[:, -1] = v_top
    V[:, 0] = v_bottom
    V[0, 0] = 0.5 * (v_bottom + v_left)
    V[-1, 0] = 0.5 * (v_bottom + v_right)
    V[0, -1] = 0.5 * (v_top + v_left)
    V[-1, -1] = 0.5 * (v_top + v_right)
    V[1:-1, 1:-1] = 0.25 * (v_bottom + v_right + v_top + v_left)
    nx = X.shape[0]
    ny = X.shape[1]
    if bc is None:
        # Explicit to prompt numba type
        bc_bool = np.array([[False]])
        bc_val = np.array([[0.0]])
    else:
        bc_bool, bc_val = bc
        for j in range(ny):
            for i in range(nx):
                if bc_bool[i, j]:
                    V[i, j] = bc_val[i, j]
    for n in range(int(Nmax)):
        Vsum = 0
        Verr = 0
        if xsym:
            for j in range(1, ny-1):
                if not bc or not bc_bool[0, j]:
                    V[0, j] = 0.25 * (V[0, j+1] + V[0, j-1] + 2*V[1, j])
        if ysym:
            for i in range(1, nx-1):
                if not bc or not bc_bool[i, 0]:
                    V[i, 0] = 0.25 * (V[i+1, 0] + V[i-1, 0] + 2*V[i, 1])
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                V_old = V[i, j]
                if not bc or not bc_bool[i, j]:
                    if dielectric is None:
                        R = 0.25 * (V[i+1, j] + V[i-1, j] +
                                    V[i, j+1] + V[i, j-1]) - V_old
                    else:
                        er_nw = dielectric[i-1, j]
                        er_ne = dielectric[i, j]
                        er_sw = dielectric[i-1, j-1]
                        er_se = dielectric[i, j-1]
                        R = (((er_sw + er_nw) * V[i-1, j] +
                              (er_nw + er_ne) * V[i, j+1] +
                              (er_ne + er_se) * V[i+1, j] +
                              (er_se + er_sw) * V[i, j-1]) /
                             (2 * (er_nw + er_ne + er_sw + er_se))) - V_old
                    V[i, j] = R * sor + V_old
                    Verr += abs(R)
                    Vsum += abs(V[i, j])
        if Vsum > 0 and Verr / Vsum < conv:
            break
    print('2D Error', Verr / Vsum, 'after', n+1, 'iterations')
    return V


@jit(nopython=True)
def poisson_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, /,
               v_left: float = 0, v_right: float = 0,
               v_top: float = 0, v_bottom: float = 0,
               v_front: float = 0, v_back: float = 0,
               dielectric: np.ndarray = None, charge: np.ndarray = None,
               bc: list = None, sor: float = 1.8,
               xsym: bool = False, ysym: bool = False, zsym: bool = False,
               conv: float = 1e-5, Nmax: int = 1e5):
    '''Three-dimension Poisson equation with fixed potential boundaries.
    Normalized charge density ps/eps can be provided via the charge argument'''
    check_arrays_3d(X, Y, Z)
    if charge is not None:
        raise Exception('Charge is currently not supported')
    # TODO enforce array types
    V = np.zeros_like(X, dtype='float64')
    V[0, :, :] = v_back
    V[-1, :, :] = v_front
    V[:, 0, :] = v_left
    V[:, -1, :] = v_right
    V[:, :, 0] = v_bottom
    V[:, :, -1] = v_top
    V[0, 0, 0] = 1/3 * (v_bottom + v_left + v_back)
    V[-1, 0, 0] = 1/3 * (v_bottom + v_left + v_front)
    V[0, -1, 0] = 1/3 * (v_bottom + v_right + v_back)
    V[-1, -1, 0] = 1/3 * (v_bottom + v_right + v_front)
    V[0, 0, -1] = 1/3 * (v_top + v_left + v_back)
    V[-1, 0, -1] = 1/3 * (v_top + v_left + v_front)
    V[0, -1, -1] = 1/3 * (v_top + v_right + v_back)
    V[-1, -1, -1] = 1/3 * (v_top + v_right + v_front)
    V[1:-1, 1:-1, 1:-1] = 1/6 * (v_bottom + v_right + v_top + v_left + v_front + v_back)
    nx = X.shape[0]
    ny = X.shape[1]
    nz = X.shape[2]
    if bc is None:
        # Explicit to prompt numba type
        bc_bool = np.array([[[False]]])
        bc_val = np.array([[[0.0]]])
    else:
        bc_bool, bc_val = bc
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if bc_bool[i, j, k]:
                        V[i, j, k] = bc_val[i, j, k]
    for n in range(int(Nmax)):
        Vsum = 0
        Verr = 0
        if xsym:
            for k in range(1, nz-1):
                for j in range(1, ny-1):
                    if not bc or not bc_bool[0, j, k]:
                        V[0, j, k] = 1/6 * (V[0, j+1, k] + V[0, j-1, k] + V[0, j, k+1] + V[0, j, k-1] + 2*V[1, j, k])
        if ysym:
            for k in range(1, nz-1):
                for i in range(1, nx-1):
                    if not bc or not bc_bool[i, 0, k]:
                        V[i, 0, k] = 1/6 * (V[i+1, 0, k] + V[i-1, 0, k] + V[i, 0, k+1] + V[i, 0, k-1] + 2*V[i, 1, k])
        if zsym:
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    if not bc or not bc_bool[i, j, 0]:
                        V[i, j, 0] = 1/6 * (V[i+1, j, 0] + V[i-1, j, 0] + V[i, j+1, 0] + V[i, j-1, 0] + 2*V[i, j, 1])
        for k in range(1, nz-1):
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    V_old = V[i, j, k]
                    if not bc or not bc_bool[i, j, k]:
                        if dielectric is None:
                            R = (V[i+1, j, k] + V[i-1, j, k] +
                                 V[i, j+1, k] + V[i, j-1, k] +
                                 V[i, j, k+1] + V[i, j, k-1]) / 6 - V_old
                        else:
                            er_brb = dielectric[i-1, j, k-1]
                            er_frb = dielectric[i, j, k-1]
                            er_blb = dielectric[i-1, j-1, k-1]
                            er_flb = dielectric[i, j-1, k-1]
                            er_brt = dielectric[i-1, j, k]
                            er_frt = dielectric[i, j, k]
                            er_blt = dielectric[i-1, j-1, k]
                            er_flt = dielectric[i, j-1, k]
                            R = (((er_brb + er_blb + er_brt + er_blt) * V[i-1, j, k] +
                                (er_frb + er_flb + er_frt + er_flt) * V[i+1, j, k] +
                                (er_blb + er_flb + er_blt + er_flt) * V[i, j-1, k] +
                                (er_brb + er_frb + er_brt + er_frt) * V[i, j+1, k] +
                                (er_brb + er_frb + er_blb + er_flb) * V[i, j, k-1] +
                                (er_brt + er_frt + er_blt + er_flt) * V[i, j, k+1]) /
                                (3 * (er_brb + er_frb + er_blb + er_flb + er_brt + er_frt + er_blt + er_flt))) - V_old
                        V[i, j, k] = R * sor + V_old
                        Verr += abs(R)
                        Vsum += abs(V[i, j, k])
        if Vsum > 0 and Verr / Vsum < conv:
            break
    print('3D Error', Verr / Vsum, 'after', n+1, 'iterations')
    return V


def gauss_1d(X: np.ndarray, V: np.ndarray, er: np.ndarray, i: int):
    '''One-dimensional Gauss' law, returning enclosed charge
    Evaluated at array index i
    Note: charge polarity is positive for V increasing with X'''
    dx = X[i+1] - X[i-1]
    return EPS0 * (er[i] * V[i+1] - er[i-1] * V[i-1] +
                   (er[i-1] - er[i]) * V[i]) / dx


def gauss_2d(X: np.ndarray, Y: np.ndarray, V: np.ndarray, er: np.ndarray,
             xi1: int, xi2: int, yi1: int, yi2: int):
    '''Two-dimensional Gauss' law, returning enclosed charge
    Evaluated along closed rectangle defined by corners:
        Bottom-left (xi1, yi1) to top-right (xi2, yi2)
    Setting xi1 to 0 implies x-symmetry, and yi0 implies y-symmetry
    In this case, the left-edge and the bottom-edge are omitted, respectively,
    with the result multiplied by 2 or 4, as appropriate
    Note: charge polarity is positive for V increasing with X or Y'''
    check_arrays_2d(X, Y)
    qe = 0
    # Top and bottom edges; dV/dy and -dV/dy, 0.5 is due to central-difference
    if yi1 == 0:
        h_edges = [(yi2, -0.5)]
    else:
        h_edges = zip((yi1, yi2), (0.5, -0.5))
    for yi, k in h_edges:
        for xi in range(xi1, xi2+1):
            qe += k * (er[xi, yi] * V[xi, yi+1] - er[xi, yi-1] * V[xi, yi-1] +
                       (er[xi, yi-1] - er[xi, yi]) * V[xi, yi])
    # Left and right edges; dV/dx and -dV/dx, 0.5 is due to central-difference
    if xi1 == 0:
        v_edges = [(xi2, -0.5)]
    else:
        v_edges = zip((xi1, xi2), (0.5, -0.5))
    for xi, k in v_edges:
        for yi in range(yi1, yi2+1):
            qe += k * (er[xi, yi] * V[xi+1, yi] - er[xi-1, yi] * V[xi-1, yi] +
                       (er[xi-1, yi] - er[xi, yi]) * V[xi, yi])
    if xi1 == 0:
        qe = 2 * qe  # TODO Do not double count point on x-axis
    if yi1 == 0:
        qe = 2 * qe  # TODO Do not double count point on y-axis
    return EPS0 * qe


def trough_analytical(X: np.ndarray, Y: np.ndarray,
                      v_left: float = 0, v_right: float = 0,
                      v_top: float = 0, v_bottom: float = 0):
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


def example_poisson_2d():
    w = 2.0
    h = 1.0
    x = np.linspace(0, w, 101)
    y = np.linspace(0, h, 51)
    X, Y = np.meshgrid(x, y, indexing='ij')
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


def example_poisson_2d_coax():
    ri = 2.0e-3
    ro = 4.0e-3
    w = 1.1 * ro
    dx = ri / 40
    Va = 10.0
    x = np.arange(0, w, dx)
    y = np.arange(-w, w, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    V = poisson_2d(X, Y, bc=bc, xsym=True)
    _, ax = plt.subplots()
    ax.contour(X, Y, V)
    ax.set_aspect('equal')
    ax.grid()
    plt.show()


def example_poisson_3d():
    ri = 2.0e-3
    ro = 4.0e-3
    w = 1.1 * ro
    dx = ri / 16
    Va = 10.0
    x = np.arange(-w, w, dx)
    y = np.arange(-w, w, dx)
    z = np.arange(-w, w, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    bc_bool = np.logical_and(R < ri, X < 0.0)
    bc_val = bc_bool * Va
    bc = (bc_bool, bc_val)
    V = poisson_3d(X, Y, Z, v_front=-Va, v_left=-Va, bc=bc)
    _, axs = plt.subplots(1, 3)
    axs[0].contour(X[:, :, int(len(z)/2)], Y[:, :, int(len(z)/2)], V[:, :, int(len(z)/2)])
    axs[0].set_title('Top View')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[1].contour(X[:, int(len(y)/2), :], Z[:, int(len(y)/2), :], V[:, int(len(y)/2), :])
    axs[1].set_title('Front View')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    axs[2].contour(Y[int(len(x)/2), :, :], Z[int(len(x)/2), :, :], V[int(len(x)/2), :, :])
    axs[2].set_title('Right View')
    axs[2].set_xlabel('y')
    axs[2].set_ylabel('z')
    for ax in axs.ravel():
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
        ax.yaxis.set_major_locator(mpl.ticker.NullLocator())
        ax.grid()
    plt.show()


if __name__ == '__main__':
    example_poisson_2d()
    example_parallel_plates()
    example_poisson_2d_coax()
    example_poisson_3d()
