#!/usr/bin/python3

'''Poisson's finite difference method'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def poisson_1d(width, x, v_left: float=0, v_right: float=0, N: int=100, charge=None):
    '''One-dimension Poisson equation with fixed potential boundaries.
    Normalized charge density ps/eps can be provided via the charge argument'''
    V = np.zeros_like(x)
    V[0] = v_left
    V[-1] = v_right
    for _ in range(N):
        V[1:-1] = 0.5 * (V[2:] + V[:-2])
        if charge is not None:
            V -= 0.5 * charge
    return V


def trough_analytical(width, height, X, Y,
                      v_bottom: float=0, v_right: float=0,
                      v_top: float=0, v_left: float=0):
    a = width
    b = height
    V = 0
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
    qp = dist(x, 0.02, 0.25 * w, 0.05 * w)
    qn = dist(x, -0.03, 0.75 * w, 0.1 * w)
    Q = (None, qp, qp + qn)

    fig, axes = plt.subplots(len(Q), 1, sharex=True)
    fig.suptitle(f'V(0)={v_left:.1f}, V({w})={v_right:.1f}')
    for i, Qi in enumerate(Q):
        V = poisson_1d(w, x, v_left, v_right, charge=Qi)
        axes[i].plot(x, V, color='b')
        if Qi is not None:
            q_axis = axes[i].twinx()
            q_axis.plot(x, Qi, color='r')
            q_axis.set_ylabel(r'$\rho$ / $\epsilon$', color='red')
        axes[i].set_ylabel('Potential (V)', color='b')
        axes[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    axes[-1].set_xlim([0, w])
    plt.show()


def example_trough():
    w = 2.0
    h = 1.0
    x = np.linspace(0, w, 101)
    y = np.linspace(0, h, 51)
    X, Y = np.meshgrid(x, y)
    V = trough_analytical(w, h, X, Y, v_top=10, v_left=5)
    plt.contour(X, Y, V)
    plt.show()


if __name__ == '__main__':
    example_poisson_1d()
    example_trough()    
