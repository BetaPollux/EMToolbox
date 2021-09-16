#!/usr/bin/python3

import cmath
import numpy as np
import pytest
from pytest import approx
from emtoolbox.tline.lossless_mtl import LosslessMtl, ZcMtl
from emtoolbox.tline.mtl_network import MtlNetwork


def pol2rect(mag, deg):
    return cmath.rect(mag, np.deg2rad(deg))


def test_network1():
    # Paul MTL P6.3
    f = 5e6
    vp = 3e8
    zc = 50
    zs = 20 - 30j
    zl = 200 + 500j
    length = 78
    vs = 50
    tline = LosslessMtl(ZcMtl(zc, vp), length)
    network = MtlNetwork(tline, zs, zl)
    sol = network.solve(f, vs)

    assert tline.n_wavelengths(f) == approx(1.3, rel=0.001)
    assert network.reflection(f) == approx(pol2rect(0.9338, 9.866), rel=0.001)
    assert network.reflection(f, 0) == approx(pol2rect(0.9338, 153.9), rel=0.001)
    assert network.input_impedance(f) == approx(pol2rect(11.73, 81.16), rel=0.001)
    assert network.get_voltage(f, sol, 0) == approx(pol2rect(20.55, 121.3), rel=0.001)
    assert network.get_voltage(f, sol, length) == approx(pol2rect(89.6, -50.45), rel=0.001)
    assert network.vswr(f) == approx(29.21, rel=0.001)


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
    tline = LosslessMtl(ZcMtl(zc))
    network = MtlNetwork(tline, zs, zl)
    assert network.vswr(1) == approx(result, rel=0.001)
