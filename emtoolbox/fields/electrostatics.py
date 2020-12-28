#!/usr/bin/python3

import numpy as np

eps0 = 8.8541878176e-12


def efield_point(q, rq, rf):
    """Calculate electric field at point 'rf' due to a single point
    charge of 'q' coulombs (C) located at point 'rq'"""
    R = rf - rq
    mag_R = np.linalg.norm(R)
    E = q / (4 * np.pi * eps0 * (mag_R ** 2))

    return E * R / mag_R


def efield_point_coll(q_coll, rq_coll, rf):
    """Calculate electric field at point 'rf' due to a collection of point
    charges of 'q[i]' coulombs (C) each located at the corresponding point
    in 'rq[i]'"""
    Et = np.zeros_like(rf)
    for i, q in enumerate(q_coll):
        R = rf - rq_coll[i]
        mag_R = np.linalg.norm(R)
        Et += q * R / (mag_R ** 3)

    return Et / (4 * np.pi * eps0)


def efield_line(ql, rq, nq, rf):
    """Calculate electric field at point 'rf' due to an infinite line of
    charge with 'ql' coulombs per meter (C/m), located at point 'rq' with
    direction unit vector 'nq' and radius """
    R = (rf - rq) - ((rf - rq).dot(nq) * nq)
    mag_R = np.linalg.norm(R)
    E = ql / (2 * np.pi * eps0 * mag_R)

    return E * R / mag_R


def efield_plane(qs, rq, nq, rf):
    """Calculate electric field at point 'rf' due to an infinite plane of
    charge with of'qs' coulombs per meter square (C/m^2) located at
    point 'rq' with normal unit vector 'nq'"""
    d = (rq - rf).dot(nq)
    E = np.sign(-d) * qs / (2 * eps0)

    return E * nq


def efield_ring(ql, rq, nq, a, rf):
    """Calculate electric field at point 'rf' due to a ring of charge
    with 'ql' coulombs per meter (C/m) centered on point 'rq' with
    normal unit vector 'nq' and radius a.
    The point 'rf' must be along the axis of the ring."""
    R = rf - rq
    if not np.array_equal(np.cross(R, nq), np.zeros_like(R)):
        raise Exception('The point rf must be along the axis of the ring.')
    mag_R = np.linalg.norm(R)
    E = ql * a * mag_R / (2 * eps0 * np.sqrt(mag_R ** 2 + a ** 2) ** 3)

    return E * R / mag_R


def efield_disk(qs, rq, nq, a, rf):
    """Calculate electric field at point 'rf' due to a disk of charge
    with 'qs' coulombs per meter square (C/m^2) centered on point 'rq'
    with normal unit vector 'nq' and radius a.
    The point 'rf' must be along the axis of the disk."""
    R = rf - rq
    if not np.array_equal(np.cross(R, nq), np.zeros_like(R)):
        raise Exception('The point rf must be along the axis of the disk.')
    mag_R = np.linalg.norm(R)
    E = qs / (2 * eps0) * (1 - mag_R / np.sqrt(mag_R ** 2 + a ** 2))

    return E * R / mag_R


if __name__ == '__main__':
    rf = np.array([-0.2, 0, -2.3])
    rq = np.array([0.2, 0.1, -2.5])
    q = 5e-9
    E = efield_point(q, rq, rf)
    print('Vector:', E)
    print('Magnitude:', np.linalg.norm(E), 'V/m')
    print('Direction:', E / np.linalg.norm(E))
