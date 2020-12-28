#!/usr/bin/python3

'''Calculate per unit length parameters of transmission lines
All functions return a dictionary of RLGC terms'''

import math
from emtoolbox.utils import em_constants as em


def coax(radius_w: float, radius_s: float, epsr: float, cond: float):
    '''A coaxial cable with:
    radius_w:   radius of wire or inner conductor (units: m)
    radius_s:   radius of shield or outer conductor (units: m)
    epsr:       relative permittivity of dielectric (units: none)
    cond:       conductivity of dielectric (units: siemens)
    '''
    k = math.log(radius_s / radius_w)
    capacitance = 2 * math.pi * em.EPS0 * epsr / k
    inductance = em.MU0 / (2 * math.pi) * k
    conductance = 2 * math.pi * cond / k
    resistance = 0

    return { 'L': inductance,
             'C': capacitance,
             'G': conductance,
             'R': resistance }
