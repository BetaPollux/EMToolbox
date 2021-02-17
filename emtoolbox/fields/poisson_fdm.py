#!/usr/bin/python3

'''Poisson's finite difference method'''

import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    w = 2.0
    h = 1.0
    x = np.linspace(0, w, 101)
    y = np.linspace(0, h, 51)
    X, Y = np.meshgrid(x, y)
    V = trough_analytical(w, h, X, Y, v_top=10, v_left=5)
    plt.contour(X, Y, V)
    plt.show()
