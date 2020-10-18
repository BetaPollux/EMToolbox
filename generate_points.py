#!/usr/bin/python3

import numpy as np


def generate_line(r0, r1, s):
    """Generate an array of 's' points equally spaced between
    points 'r0' and 'r1'. Minimum 2 points."""
    if s < 2:
        raise Exception('s cannot be < 2')
    R = r1 - r0
    t = np.linspace(0.0, 1.0, s)
    points = np.zeros((s, len(R)))
    for i, ti in enumerate(t):
        points[i] = r0 + ti * R

    return points


def generate_ring(a, r0, n, s):
    """Generate an array of 's' points equally spaced along a ring
    of radius 'a', centered on 'r0' with normal 'n'. Minimum 2 points."""
    if s < 2:
        raise Exception('s cannot be < 2')
    t = np.linspace(0.0, 2 * np.pi, s, endpoint=False)
    points = np.zeros((s, len(r0)))
    for i, ti in enumerate(t):  # Generate ring in x-y plane (n = [0,0,1])
        points[i] = a * np.array([np.cos(ti), np.sin(ti), 0.0])
    # Rotate to match n (symmetrical for z rotation)
    tx = np.arcsin(n[1])  # FIXME this is not correct!
    Rx = np.array([[1.0,        0.0,         0.0],
                   [0.0,        np.cos(tx),  np.sin(tx)],
                   [0.0,        -np.sin(tx), np.cos(tx)]])
    ty = np.arcsin(n[0])  # FIXME this is not correct!
    Ry = np.array([[np.cos(ty),  0.0,        np.sin(ty)],
                   [0.0,         1.0,        0.0],
                   [-np.sin(ty), 0.0,        np.cos(ty)]])
    # Rotate then translate to r0
    for i, pt in enumerate(points):
        points[i] = pt @ Rx @ Ry
        points[i] += r0

    return points
