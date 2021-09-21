#!/usr/bin/python3

'''Test transmission line TLine module'''

import numpy as np
import pytest
from pytest import approx
from emtoolbox.tline.tline import TLine
import emtoolbox.utils.constants as const

FREQ = 100e6
ZC = 50
ER = 2.0
LENGTH = 1.0


def char_impedance():
    tline = TLine.create_lossless(FREQ, ZC, ER, LENGTH)
    assert tline.impedance() == approx(ZC, rel=0.001)


def test_lossless():
    line = TLine.create_lossless(FREQ, ZC, ER, LENGTH)
    assert np.sqrt(line.L / line.C) == approx(ZC)
    assert line.R == 0
    assert line.G == 0
    assert line.attn_const() == 0


def test_lowloss():
    R = 2.0
    G = 1e-9 * FREQ
    line = TLine.create_lowloss(FREQ, ZC, ER, LENGTH, R=R, G=G)
    assert np.sqrt(line.L / line.C) == approx(ZC)
    assert line.R == R
    assert line.G == G
    assert line.attn_const() > 0.1


def test_velocity():
    line = TLine.create_lossless(FREQ, ZC, ER, LENGTH)
    assert line.velocity() == approx(const.VP0 / np.sqrt(ER), rel=1e-3)


def test_delay():
    line = TLine.create_lossless(FREQ, ZC, ER, LENGTH)
    vp = const.VP0 / np.sqrt(ER)
    assert line.delay() == approx(line.length / vp, rel=1e-3)


@pytest.mark.parametrize(
    "length, er, result",
    [
        (1.0, 1.0, 3.333e-9),
        (1.0, 4.0, 6.67e-9),
        (2.0, 1.0, 6.667e-9),
        (10.0, 1.0, 33.33e-9),
        (10.0, 9.0, 100e-9)
    ],
)
def test_delay(length, er, result):
    line = TLine.create_lossless(FREQ, ZC, er, length)
    assert line.delay() == approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "freq, er, result",
    [
        (300e6, 1.0, 1.0),
        (600e6, 1.0, 0.5),
        (3e9, 1.0, 0.1),
        (300e6, 4.0, 0.5)
    ],
)
def test_wavelength(freq, er, result):
    tline = TLine.create_lossless(freq, ZC, er, LENGTH)
    assert tline.wavelength() == approx(result, rel=0.001)


@pytest.mark.parametrize(
    "freq, result",
    [
        (300e6, 1.0),
        (600e6, 2.0),
        (3e9, 10.0)
    ],
)
def test_n_wavelengths(freq, result):
    tline = TLine.create_lossless(freq, ZC, 1.0, LENGTH)
    assert tline.n_wavelengths() == approx(result, rel=0.001)


def test_phase_const():
    line = TLine.create_lossless(FREQ, ZC, ER, LENGTH)
    wavelength = const.VP0 / np.sqrt(ER) / FREQ
    assert line.phase_const() == approx(2 * np.pi / wavelength, rel=1e-3)


def test_attn():
    zc = 63.8
    R = 25.46
    G = 1.423e-2
    line = TLine.create_lowloss(FREQ, zc, ER, LENGTH, R=R, G=G)
    assert line.attn_const() == approx(0.651, rel=1e-3)


def test_lossless_multf():
    freq = np.array([1e6, 10e6, 100e6])
    line = TLine.create_lossless(freq, ZC, ER, LENGTH)
    assert np.sqrt(line.L / line.C) == approx(ZC)
    assert line.R == 0
    assert line.G == 0
    assert line.attn_const() == approx(0)


@pytest.mark.parametrize(
    "freq, result",
    [
        (3e6, 2 * np.pi / 100.0),
        (30e6, 2 * np.pi / 10.0),
        (300e6, 2 * np.pi)
    ],
)
def test_phase_const(freq, result):
    tline = TLine.create_lossless(freq, ZC, 1.0)
    assert tline.phase_const() == approx(result, rel=0.001)
    assert tline.prop_const() == approx(1.j * result, rel=0.001)


@pytest.mark.parametrize(
    "n, result",
    [
        (0.25, np.array([[0, -1.j*ZC], [-1.j/ZC, 0]])),
        (0.5, np.array([[-1, 0], [0, -1]])),
        (0.75, np.array([[0, 1.j*ZC], [1.j/ZC, 0]])),
        (1.0, np.array([[1, 0], [0, 1]]))
    ],
)
def test_chain_param(n, result):
    freq = 30e6
    length = n * const.VP0 / freq
    tline = TLine.create_lossless(freq, ZC, 1.0, length=length)
    assert tline.chain_param() == approx(result, rel=0.001)
