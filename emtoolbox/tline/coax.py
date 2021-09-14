'''Coaxial cable functions'''

import math
from emtoolbox.utils.constants import MU0, EPS0

# TODO remove **_ and fix calling code


def impedance(radius_w: float, radius_s: float, epsr: float, **_) -> float:
    '''Calculate characteristic impedance, assuming a DC lossless line
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
        epsr:           relative permittivity of dielectric (units: none)
    Return:
        impedance (units: ohms)'''
    return math.sqrt(MU0 / (EPS0 * epsr)) * math.log(radius_s / radius_w) / (2 * math.pi)


def inductance(radius_w: float, radius_s: float, **_) -> float:
    '''Calculate inductance per unit length.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
    Return:
        inductance (units: henries/meter)'''
    return MU0 / (2 * math.pi) * math.log(radius_s / radius_w)


def capacitance(radius_w: float, radius_s: float, epsr: float, **_) -> float:
    '''Calculate capacitance per unit length.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
        epsr:           relative permittivity of dielectric (units: none)
    Return:
        capacitance (units: farads/meter)'''
    return 2 * math.pi * EPS0 * epsr / math.log(radius_s / radius_w)


def conductance_simple(radius_w: float, radius_s: float, cond_d: float, **_):
    '''Calculate conductance per unit length
    Dielectricl model is a simple conductor.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
        cond_d:         conductivity of dielectric (units: siemens/meter)
    Return:
        conductance (units: siemens/meter)'''
    return 2 * math.pi * cond_d / math.log(radius_s / radius_w)


def conductance_loss_tangent(radius_w: float, radius_s: float,
                             epsr: float, loss_tangent: float, **_):
    '''Calculate conductance per unit length.
    Dielectric model is a constant loss tangent.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        radius_s:       radius of shield or outer conductor (units: m)
        epsr:           relative permittivity of dielectric (units: none)
        loss_tangent:   loss tangent (units: none)
    Return:
        conductance(w)  function of w (units: siemens/meter)'''
    return lambda w: 2 * math.pi * w * loss_tangent * EPS0 * epsr / math.log(radius_s / radius_w)


def resistance_dc(radius_w: float, cond_c: float, **_):
    '''Calculate resistance per unit length, assuming DC.
    Neglects resistance of the shield conductor.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        cond_c:         conductivity of conductors (units: siemens/meter)
    Return:
        resistance (units: ohm/meter)'''
    return 1.0 / (math.pi * cond_c * radius_w ** 2)


def resistance_skin_effect(radius_w: float, cond_c: float, **_):
    '''Calculate resistance per unit length, assuming simple skin effect.
    Neglects resistance of the shield conductor.
    Arguments:
        radius_w:       radius of wire or inner conductor (units: m)
        cond_c:         conductivity of conductors (units: siemens/meter)
    Return:
        resistance(w)   function of w (units: ohm/meter)'''
    # TODO invalid at low frequency
    return lambda w: math.sqrt((w * MU0) / (2 * math.pi ** 2 * cond_c)) / (2 * radius_w)


if __name__ == '__main__':
    RW = 1e-3
    RS = 3e-3
    EPSR = 2
    L = inductance(RW, RS)
    C = capacitance(RW, RS, EPSR)
    ZC = impedance(RW, RS, EPSR)
    print(f'Radii {RW}, {RS}')
    print(f'EPSR {EPSR}')
    print(f'{L:.3e} H/m, {C:.3e} F/m')
    print(f'ZC {ZC:.3f}, or sqrt(L/C) {math.sqrt(L/C):.3f}')
