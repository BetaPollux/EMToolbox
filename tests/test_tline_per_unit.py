#!/usr/bin/python3


'''Test the transmission line per unit length functions'''

from pytest import approx
import emtoolbox.tline.tline_per_unit as pul
from emtoolbox.utils.conversions import meters_from_mils

def test_coax():
    '''Test case based on CR Paul Multiconductor Transmission Lines
    Exercise 4.10'''
    radius_w = meters_from_mils(20.15)
    radius_s = meters_from_mils(90)
    epsr = 1.45
    coax = pul.coax(radius_w, radius_s, epsr)

    assert coax['L'] == approx(0.2993e-6, rel=1e-3)
    assert coax['C'] == approx(53.83e-12, rel=1e-3)
