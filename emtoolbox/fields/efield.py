#!/usr/bin/python3

import numpy as np


def efield_1d(potential, dx: float):
    E = np.zeros_like(potential)
    E[0] = (potential[0] - potential[1]) / dx
    E[-1] = (potential[-2] - potential[-1]) / dx
    E[1:-1] = 0.5 / dx * (potential[:-2] - potential[2:])
    return E


def efield_2d(potential, dx: float, dy: float):
    Ex = np.zeros_like(potential)
    Ey = np.zeros_like(potential)
    Ex[:, 0] = (potential[:, 0] - potential[:, 1]) / dx
    Ex[:, -1] = (potential[:, -2] - potential[:, -1]) / dx
    Ex[:, 1:-1] = 0.5 / dx * (potential[:, :-2] - potential[:, 2:])

    Ey[0, :] = (potential[:, 0] - potential[:, 1]) / dy
    Ey[-1, :] = (potential[:, -2] - potential[:, -1]) / dy
    Ey[1:-1, :] = 0.5 / dy * (potential[:-2, :] - potential[2:, :])

    return Ex, Ey
