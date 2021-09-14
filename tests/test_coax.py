#!/usr/bin/python3

'''Test TLine Coax functions'''

import math
from pytest import approx
import emtoolbox.tline.coax as coax
import emtoolbox.utils.conversions as convert
import emtoolbox.utils.constants as const

FREQ = 100e6
W = 2 * math.pi * FREQ
EPSR = 2.3
RADIUS_W = convert.meters_from_mils(16)
RADIUS_S = convert.meters_from_mils(58)


def test_impedance():
    zc = coax.impedance(RADIUS_W, RADIUS_S, EPSR)
    assert zc == approx(50.0, abs=1.0)


def test_inductance():
    ind = coax.inductance(RADIUS_W, RADIUS_S)
    assert ind == approx(0.2576e-6, rel=1e-3)


def test_capacitance():
    cap = coax.capacitance(RADIUS_W, RADIUS_S, EPSR)
    assert cap == approx(99.2e-12, rel=1e-3)


def test_conductance_simple():
    sigma = 10
    con = coax.conductance_simple(RADIUS_W, RADIUS_S, sigma)
    cap = coax.capacitance(RADIUS_W, RADIUS_S, EPSR)
    assert con == approx(sigma / (const.EPS0 * EPSR) * cap, rel=1e-3)


def test_conductance_loss_tangent():
    tan = 0.02
    con = coax.conductance_loss_tangent(RADIUS_W, RADIUS_S, EPSR, tan)
    cap = coax.capacitance(RADIUS_W, RADIUS_S, EPSR)
    assert con(W) == approx(W * tan * cap, rel=1e-3)


def test_conductance_loss_tangent_mult():
    tan = 0.02
    con1 = coax.conductance_loss_tangent(RADIUS_W, RADIUS_S, EPSR, tan)
    con2 = coax.conductance_loss_tangent(RADIUS_W, 2 * RADIUS_S, EPSR, tan)
    assert con1(W) != con2(W)


def test_resistance_dc():
    rad = convert.wire_radius_awg(28)
    res = coax.resistance_dc(rad, const.COND_CU)
    assert res == approx(0.213, rel=1e-3)


def test_resistance_skin_effect():
    res = coax.resistance_skin_effect(RADIUS_W, const.COND_CU)
    assert res(W) == approx(1.022, rel=1e-3)


def test_resistance_skin_effect_mult():
    res1 = coax.resistance_skin_effect(RADIUS_W, const.COND_CU)
    res2 = coax.resistance_skin_effect(2 * RADIUS_W, const.COND_CU)
    assert res1(W) != res2(W)
