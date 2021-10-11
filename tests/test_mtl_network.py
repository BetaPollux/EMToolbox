#!/usr/bin/python3

import cmath
import numpy as np
import pytest
from pytest import approx
from emtoolbox.tline.tline import TLine
from emtoolbox.tline.mtl_network import MtlNetwork


def pol2rect(mag, deg):
    return cmath.rect(mag, np.deg2rad(deg))


@pytest.mark.parametrize(
    "f",
    [
        5e6,
        np.array([5e6]),
        np.array([5e6, 5e6]),
        np.array([5e6, 5e6, 5e6]),
    ],
)
def test_network1_simple(f):
    # Paul MTL P6.3
    vp = 3e8
    zc = 50
    zs = 20 - 30j
    zl = 200 + 500j
    length = 78
    tline = TLine.create_lowloss(zc, freq=f, vp=vp, length=length)
    network = MtlNetwork(tline, zs, zl)
    assert tline.n_wavelengths() == approx(1.3, rel=0.001)
    assert network.reflection() == approx(pol2rect(0.9338, 9.866), rel=0.001)
    assert network.reflection(0) == approx(pol2rect(0.9338, 153.9), rel=0.001)
    assert network.input_impedance() == approx(pol2rect(11.73, 81.16), rel=0.001)


@pytest.mark.parametrize(
    "f",
    [
        5e6,
        np.array([5e6]),
        np.array([5e6, 5e6]),
        np.array([5e6, 5e6, 5e6]),
    ],
)
def test_network1_solve(f):
    # Paul MTL P6.3
    vp = 3e8
    zc = 50
    zs = 20 - 30j
    zl = 200 + 500j
    length = 78
    vs = 50
    tline = TLine.create_lowloss(zc, freq=f, vp=vp, length=length)
    network = MtlNetwork(tline, zs, zl)
    sol = network.solve(vs)

    assert network.get_voltage(sol, 0) == approx(pol2rect(20.55, 121.3), rel=0.001)
    assert network.get_voltage(sol, length) == approx(pol2rect(89.6, -50.45), rel=0.001)
    assert network.vswr() == approx(29.21, rel=0.001)


@pytest.mark.parametrize(
    "zl, result",
    [
        (50, 1.0),
        (100, 2.0),
        (10, 5.0),
        (0, np.inf)
    ],
)
def test_vswr(zl, result):
    zc = 50
    zs = 50
    tline = TLine.create_lowloss(zc)
    network = MtlNetwork(tline, zs, zl)
    assert network.vswr() == approx(result, rel=0.001)
