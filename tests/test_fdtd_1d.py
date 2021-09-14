#!#!/bin/usr/python3

import numpy as np
import pytest
from pytest import approx
import emtoolbox.shielding.planar_shield as ps
from emtoolbox.utils.constants import ETA0, COND_CU
import emtoolbox.fdtd.fdtd_1d as fdtd
import emtoolbox.shielding.planar_shield as shield


@pytest.mark.parametrize(
    "er, cond, f",
    [   # TODO accuracy issues with certain cases
        (1.0, 0.0, 10e6),
        (4.0, 0.0, 10e6),
        # (16.0, 0.0, 10e6),
        (4.0, 0.04, 10e6),
        (4.0, 0.04, 50e6),
        # (4.0, 0.04, 200e6),
        # (1.0, 0.5, 10e6)
    ],
)
def test_dielectric(er, cond, f):
    thk = 0.5
    total_time = max(1.0 / f, 40e-9)
    dx = 0.005
    grid = fdtd.Grid1D(dx, 2.0)
    print(grid)
    grid.set_material(1.0, 1.0 + thk, er=er, cond=cond)

    amplitude = 1.0
    source = fdtd.Sinusoid(0.1, 'Sine', amplitude, f)
    grid.add_source(source)
    transmission_probe = fdtd.Probe(1.9)
    grid.add_probe(transmission_probe)
    grid.solve(total_time)
    result = transmission_probe.data.max()

    air = shield.Material()
    dielectric = shield.Material(er=er, cond=cond, thickness=thk)
    expected = 1 / shield.shielding_effectiveness(f, air, dielectric)

    assert result == approx(expected, rel=0.01)
