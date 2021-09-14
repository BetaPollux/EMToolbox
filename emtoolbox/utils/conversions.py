#!/usr/bin/python3

'''Unit conversion functions'''

import math
import re


def meters_from_mils(pos: float) -> float:
    '''Convert mils (0.001 inch) to meters'''
    return 2.54e-5 * pos


def wire_diameter_awg(awg: int) -> float:
    '''Gets a wire diameter, in meters, from AWG'''
    diameter = {
        '8': 3.264e-3,
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


def engstr(num: float, precision: int = 3) -> str:
    '''Convert a number into an engineering-formatted string'''
    prefixes = {
        -15: 'f',
        -12: 'p',
        -9: 'n',
        -6: 'u',
        -3: 'm',
        0: '',
        3: 'k',
        6: 'M',
        9: 'G',
        12: 'T',
        15: 'P'
    }
    if num == 0:
        exponent = 0
    elif abs(num) > 1:
        exponent = 3 * int(math.log10(abs(num)) / 3)
    else:
        exponent = 3 * (int(math.log10(abs(num)) / 3) - 1)
    try:
        return f'{num/10**exponent:.{precision}f}{prefixes[exponent]}'
    except KeyError:
        return f'{num:.{precision}e}'


def engfloat(num: str) -> float:
    '''Convert an engineering-formatted string into a float'''
    prefixes = {
        'f': 1e-15,
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'm': 1e-3,
        '': 1.0,
        'k': 1e3,
        'M': 1e6,
        'G': 1e9,
        'T': 12.0,
        'P': 15.0
    }
    regexp = re.match(r'^([0-9\-+eE.]+)\s*(\w*)$', num)
    return float(regexp.groups()[0]) * prefixes[regexp.groups()[1]]
