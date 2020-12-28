#!/usr/bin/python3

'''Calculate per unit length parameters of transmission lines
All functions return a dictionary of RLGC terms'''

import math
from emtoolbox.utils.constants import MU0, EPS0


def coax(radius_w: float, radius_s: float, epsr: float,
         cond: float = None, loss_tangent: float = 0):
    '''A coaxial cable with:
    radius_w:       radius of wire or inner conductor (units: m)
    radius_s:       radius of shield or outer conductor (units: m)
    cond:           conductivity of wire and shield (units: S/m)
    epsr:           relative permittivity of dielectric (units: none)
    loss_tangent:   loss tangent of dielectric (units: none)
    '''
    k = math.log(radius_s / radius_w)
    capacitance = 2 * math.pi * EPS0 * epsr / k
    inductance = MU0 / (2 * math.pi) * k
    conductance = lambda w: w * loss_tangent * capacitance
    if cond:
        resistance = 1.0 / (cond * math.pi * radius_w ** 2)
    else:
        resistance = 0.0

    return { 'L': inductance,
             'C': capacitance,
             'G': conductance,
             'R': resistance }
