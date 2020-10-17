#!/usr/bin/python3

import numpy as np

eps0 = 8.8541878176e-12


def efield_point(q, rq, rf):
    R = rf - rq
    mag_R = np.linalg.norm(R)
    E = q / (4 * np.pi * eps0 * (mag_R ** 2))

    return E * R / mag_R


def efield_point_coll(q_coll, rq_coll, rf):
    Et = np.zeros_like(rf)
    for i, q in enumerate(q_coll):
        R = rf - rq_coll[i]
        mag_R = np.linalg.norm(R)
        Ei = q * R / (mag_R ** 3)
        Et += Ei

    E = Et / (4 * np.pi * eps0)

    return E


def efield_line(ql, rq, nq, rf):
    R = (rf - rq) - ((rf - rq).dot(nq) * nq)
    mag_R = np.linalg.norm(R)
    E = ql / (2 * np.pi * eps0 * mag_R)

    return E * R / mag_R


def efield_plane(qs, rq, nq, rf):
    d = (rq - rf).dot(nq)
    E = np.sign(-d) * qs / (2 * eps0)

    return E * nq


if __name__ == '__main__':
    rf = np.array([-0.2, 0, -2.3])
    rq = np.array([0.2, 0.1, -2.5])
    q = 5e-9
    E = efield_point(q, rq, rf)
    print('Vector:', E)
    print('Magnitude:', np.linalg.norm(E), 'V/m')
    print('Direction:', E / np.linalg.norm(E))
