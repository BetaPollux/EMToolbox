#!/bin/usr/python3

import numpy as np
import pytest
from pytest import approx
from emtoolbox.utils.constants import VP0
from emtoolbox.tline.wire_mtl import WireMtl
from emtoolbox.tline.lossless_mtl import LosslessMtl

ZC = 50

class ZcMtl():
    def __init__(self, zc, er=1.0):
        self.velocity = VP0 / np.sqrt(er)
        self.L = zc / self.velocity
        self.C = 1 / (zc * self.velocity)
    
    def inductance(self):
        return np.array([[self.L]])
    
    def capacitance(self):
        return np.array([[self.C]])


@pytest.mark.parametrize(
    "freq, result",
    [
        (300e6, 1.0),
        (600e6, 0.5),
        (3e9, 0.1)
    ],
)
def test_wavelength(freq, result):
    tline = LosslessMtl(ZcMtl(ZC), length=1.0)
    assert tline.wavelength(freq) == approx(result, rel=0.001)

@pytest.mark.parametrize(
    "freq, result",
    [
        (300e6, 1.0),
        (600e6, 2.0),
        (3e9, 10.0)
    ],
)
def test_n_wavelengths(freq, result):
    tline = LosslessMtl(ZcMtl(ZC), length=1.0)
    assert tline.n_wavelengths(freq) == approx(result, rel=0.001)

def test_velocity():
    assert LosslessMtl(ZcMtl(ZC)).velocity() == approx(VP0)

@pytest.mark.parametrize(
    "length, result",
    [
        (1.0, 3.333e-9),
        (2.0, 6.667e-9),
        (10.0, 33.33e-9)
    ],
)
def test_delay(length, result):
    tline = LosslessMtl(ZcMtl(ZC), length=length)
    assert tline.delay() == approx(result, rel=0.001)

def char_impedance():
    tline = LosslessMtl(ZcMtl(ZC))
    assert tline.char_impedance() == approx(ZC, rel=0.001)

@pytest.mark.parametrize(
    "freq, result",
    [
        (3e6, 2 * np.pi / 100.0),
        (30e6, 2 * np.pi / 10.0),
        (300e6, 2 * np.pi)
    ],
)
def test_phase_const(freq, result):
    tline = LosslessMtl(ZcMtl(ZC))
    assert tline.phase_const(freq) == approx(result, rel=0.001)
    assert tline.prop_const(freq) == approx(1.j * result, rel=0.001)

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
    length = n * VP0 / freq
    tline = LosslessMtl(ZcMtl(ZC), length=length)
    assert tline.chain_param(freq) == approx(result, rel=0.001)
