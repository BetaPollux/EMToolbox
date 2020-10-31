#! /usr/bin/python3

import numpy as np

mu0 = 4e-7 * np.pi
eps0 = 8.854e-12


def meters_from_mils(pos):
    return 2.54e-5 * pos


class WideSep_Wire():
    def self_inductance(pos0, radius0, pos1, radius1):
        d10 = np.linalg.norm(pos1 - pos0)
        ls = mu0 / (2 * np.pi) * np.log(d10 * d10 / (radius1 * radius0))

        return ls

    def mutual_inductance(pos0, radius0, pos1, pos2):
        d10 = np.linalg.norm(pos1 - pos0)
        d20 = np.linalg.norm(pos2 - pos0)
        d21 = np.linalg.norm(pos2 - pos1)
        lm = mu0 / (2 * np.pi) * np.log(d10 * d20 / (d21 * radius0))

        return lm


class WideSep_Plane():
    def self_inductance(pos, radius):
        h = pos[1]
        ls = mu0 / (2 * np.pi) * np.log(2 * h / radius)

        return ls

    def mutual_inductance(pos1, pos2):
        s = np.linalg.norm(pos2 - pos1)
        h1 = pos1[1]
        h2 = pos2[1]
        lm = mu0 / (4 * np.pi) * np.log(1 + (4 * h1 * h2) / (s ** 2))

        return lm


class WideSep_Shield():
    def self_inductance(radius_sh, pos_r, radius_w):
        a = (radius_sh ** 2 - pos_r ** 2) / (radius_sh * radius_w)
        ls = mu0 / (2 * np.pi) * np.log(a)

        return ls

    def mutual_inductance(radius_sh, pos_r1, pos_phi1, pos_r2, pos_phi2):
        n = (pos_r1 * pos_r2) ** 2 + radius_sh ** 4
        n += -2 * pos_r1 * pos_r2 * radius_sh ** 2 * (
                np.cos(pos_phi2 - pos_phi1))
        d = (pos_r1 * pos_r2) ** 2 + pos_r2 ** 4
        d += -2 * pos_r1 * pos_r2 ** 3 * np.cos(pos_phi2 - pos_phi1)

        lm = mu0 / (2 * np.pi) * np.log(pos_r2 / radius_sh * np.sqrt(n/d))

        return lm


class Mtl():
    def assemble_inductance_matrix(ls, lm):
        L = np.diag(ls)
        for i, lmi in enumerate(lm):
            L[i, i + 1] = lmi
            L[i + 1, i] = lmi

        return L

    def generate_capacitance_matrix(inductance_matrix, er):
        C = mu0 * eps0 * er * np.linalg.inv(inductance_matrix)

        return C


if __name__ == '__main__':
    pos = np.array([[50, 0],
                    [0, 0],
                    [100, 0]])

    wire_radius = np.array([7.5,
                            7.5,
                            7.5])

    p = meters_from_mils(pos)

    r = meters_from_mils(wire_radius)
    lG = WideSep_Wire.self_inductance(p[0], r[0], p[1], r[1])
    lR = WideSep_Wire.self_inductance(p[0], r[0], p[2], r[2])
    lM = WideSep_Wire.mutual_inductance(p[0], r[0], p[1], p[2])
    L = Mtl.assemble_inductance_matrix([lG, lR], [lM])
    C = Mtl.generate_capacitance_matrix(L, 1.0)

    print('lG', lG)
    print('lR', lR)
    print('lM', lM)
    print('L', L)
    print('C', C)
