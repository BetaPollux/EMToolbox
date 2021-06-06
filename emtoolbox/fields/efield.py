#!/usr/bin/python3

import numpy as np


def efield_1d(potential, dx: float):
    return -1/dx * np.gradient(potential)


def efield_2d(potential, dx: float, dy: float):
    Ex, Ey = np.gradient(potential)
    Ex *= -1 / dx
    Ey *= -1 / dy
    return Ex, Ey
