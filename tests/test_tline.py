#!/usr/bin/python3

'''Test transmission line TLine module'''

import math
from pytest import approx
from emtoolbox.tline.tline import TLine
import emtoolbox.utils.constants as const

FREQ = 100e6
W = 2 * math.pi * FREQ
ZC = 50
ER = 2.0
LENGTH = 1.0


def test_lossless():
    line = TLine.create_lossless(ZC, ER, LENGTH)
    assert math.sqrt(line.l / line.c) == approx(ZC)
    assert line.r(W) == 0
    assert line.g(W) == 0
    assert line.attn_const(W) == 0


def test_lowloss():
    r = 2.0
    g = lambda w: 1e-9 * w
    line = TLine.create_lowloss(ZC, ER, LENGTH, r=r, g=g)
    assert math.sqrt(line.l / line.c) == approx(ZC)
    assert line.r(W) == r
    assert line.g(W) == g(W)
    assert line.attn_const(W) > 0.1


def test_velocity():
    line = TLine.create_lossless(ZC, ER, LENGTH)
    assert line.velocity(W) == approx(const.VP0 / math.sqrt(ER), rel=1e-3)


def test_delay():
    line = TLine.create_lossless(ZC, ER, LENGTH)
    assert line.delay(W) == approx(line.length / line.velocity(W), rel=1e-3)


def test_phase():
    line = TLine.create_lossless(ZC, ER, LENGTH)
    assert line.phase_const(W) == approx(W / line.velocity(W), rel=1e-3)


def test_attn():
    zc = 63.8
    r = 25.46
    g = 1.423e-2
    line = TLine.create_lowloss(zc, ER, LENGTH, r=r, g=g)
    assert line.attn_const(W) == approx(0.651, rel=1e-3)
