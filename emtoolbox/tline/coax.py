'''Coaxial cable functions'''

import math
from emtoolbox.utils.constants import MU0, EPS0


def impedance_simple(radius_w: float, radius_s: float, epsr: float) -> float:
    '''Calculate characteristic impedance, assuming a DC lossless line
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
        epsr:           relative permittivity of dielectric (units: none)
    Return:
        impedance (units: ohms)'''
    return math.sqrt(MU0 / (EPS0 * epsr)) * math.log(radius_s / radius_w) / (2 * math.pi)

def inductance(radius_w: float, radius_s: float) -> float:
    '''Calculate inductance per unit length.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
    Return:
        inductance (units: henries/meter)'''
    return MU0 / (2 * math.pi) * math.log(radius_s / radius_w)

def capacitance(radius_w: float, radius_s: float, epsr: float) -> float:
    '''Calculate capacitance per unit length.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
        epsr:           relative permittivity of dielectric (units: none)
    Return:
        capacitance (units: farads/meter)'''
    return 2 * math.pi * EPS0 * epsr / math.log(radius_s / radius_w)

def conductance_simple(radius_w: float, radius_s: float, cond: float):
    '''Calculate conductance per unit length
    Dielectricl model is a simple conductor.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
        cond:           conductivity of dielectric (units: siemens/meter)
    Return:
        conductance (units: siemens/meter)'''
    return 2 * math.pi * cond / math.log(radius_s / radius_w)

def conductance_loss_tangent(radius_w: float, radius_s: float,
                             epsr: float, loss_tangent: float, w: float):
    '''Calculate conductance per unit length.
    Dielectric model is a constant loss tangent.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
        epsr:           relative permittivity of dielectric (units: none)
        loss_tangent:   loss tangent (units: none)
        w:              angular frequency (2pi * f)
    Return:
        conductance (units: siemens/meter)'''
    return 2 * math.pi * w * loss_tangent * EPS0 * epsr / math.log(radius_s / radius_w)

def resistance_dc(radius_w: float, cond: float):
    '''Calculate resistance per unit length, assuming DC.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        cond:           conductivity of wire (units: siemens/meter)
    Return:
        conductance (units: siemens/meter)'''
    return 1.0 / (math.pi * cond * radius_w ** 2)

def resistance_skin_effect(radius_w: float, cond: float, w: float):
    '''Calculate resistance per unit length, assuming simple skin effect
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        cond:           conductivity of wire (units: siemens/meter)
        w:              angular frequency (2pi * f)
    Return:
        conductance (units: siemens/meter)'''
    skin_depth = math.sqrt(2 / (cond * w * MU0))
    return 1.0 / (2 * math.pi * radius_w * cond * skin_depth)

if __name__ == '__main__':
    RW = 1e-3
    RS = 3e-3
    EPSR = 2
    L = inductance(RW, RS)
    C = capacitance(RW, RS, EPSR)
    ZC = impedance_simple(RW, RS, EPSR)
    print(f'Radii {RW}, {RS}')
    print(f'EPSR {EPSR}')
    print(f'{L:.3e} H/m, {C:.3e} F/m')
    print(f'ZC {ZC:.3f}, or sqrt(L/C) {math.sqrt(L/C):.3f}')
