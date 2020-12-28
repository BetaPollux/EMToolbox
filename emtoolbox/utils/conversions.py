#!/usr/bin/python3

'''Unit conversion functions'''

def meters_from_mils(pos: float) -> float:
    '''Convert mils (0.001 inch) to meters'''
    return 2.54e-5 * pos


def wire_diameter_awg(awg: int) -> float:
    '''Gets a wire diameter, in meters, from AWG'''
    diameter = {    '8': 3.264e-3,
                    '10': 2.906e-3,
                    '12': 2.053e-3,
                    '14': 1.628e-3,
                    '16': 1.291e-3,
                    '18': 1.024e-3,
                    '20': 0.812e-3,
                    '22': 0.644e-3,
                    '24': 0.511e-3,
                    '26': 0.405e-3,
                    '28': 0.321e-3,
                    '30': 0.255e-3,
                    '32': 0.202e-3,
                    '34': 0.160e-3,
                    '36': 0.127e-3,
                    '38': 0.101e-3,
                    '40': 0.0799e-3,
                }
    return diameter[str(awg)]


def wire_radius_awg(awg: int) -> float:
    '''Gets a wire radius, in meters, from AWG'''
    return 0.5 * wire_diameter_awg(awg)
