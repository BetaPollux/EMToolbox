#!/usr/bin/python3

import numpy as np
import pytest
from pytest import approx
import emtoolbox.shielding.planar_shield as ps
from emtoolbox.utils.constants import ETA0, COND_CU


@pytest.mark.parametrize(
    "er, ur, cond, f, result",
    [
        (1.0, 1.0, 0.0, 1e6, ETA0),
        (4.0, 1.0, 0.0, 1e6, 0.5 * ETA0),
        (1.0, 4.0, 0.0, 1e6, 2.0 * ETA0)
    ],
)
def test_impedance(er, ur, cond, f, result):
    mat = ps.Material(er=er, ur=ur, cond=cond)
    assert mat.impedance(f) == approx(result)


@pytest.mark.parametrize(
    "ur, cond_r, result",
    [
        # Paul EMC, Review Ex 10.1
        (1.0, 0.61, 106),
        (1.0, 0.26, 102),
        (500.0, 0.02, 64)
    ],
)
def test_reflection(ur, cond_r, result):
    f = 1e6
    air = ps.Material()
    shld = ps.Material(ur=ur, cond=(cond_r * COND_CU))
    se_db = ps.db(ps.reflection_loss(f, air, shld))
    assert se_db == approx(result, rel=0.01)


@pytest.mark.parametrize(
    "ur, cond_r, result",
    [
        # Paul EMC, Review Ex 10.2
        (1.0, 0.61, 8.4582e-5),
        (1.0, 0.26, 1.2954e-4),
        (500.0, 0.02, 2.0828e-5)
    ],
)
def test_skin_depth(ur, cond_r, result):
    f = 1e6
    shld = ps.Material(ur=ur, cond=(cond_r * COND_CU))
    assert shld.skin_depth(f) == approx(result, rel=0.01)


@pytest.mark.parametrize(
    "ur, cond_r, result",
    [
        # Paul EMC, Review Ex 10.3
        (1.0, 0.61, 326),
        (1.0, 0.26, 213),
        (500.0, 0.02, 1320)
    ],
)
def test_absorption(ur, cond_r, result):
    f = 1e6
    t = 3.175e-3
    shld = ps.Material(ur=ur, cond=(cond_r * COND_CU), thickness=t)
    se_db = ps.db(ps.absorption_loss(f, shld))
    assert se_db == approx(result, rel=0.01)
