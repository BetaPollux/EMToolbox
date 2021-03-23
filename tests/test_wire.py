#!/usr/bin/python3

import numpy as np
import pytest
from pytest import approx
from emtoolbox.tline.wire import Wire, Plane, Shield


def test_radius_zero():
    with pytest.raises(ValueError):
        Wire(0, 0, 0)


def test_distance():
    a = Wire(-1, 0, 0.5)
    b = Wire(1, 0, 0.5)
    assert a.distance_to(b) == approx(2.0)


def test_distance_diag():
    a = Wire(-1, -1, 0.5)
    b = Wire(1, 1, 0.5)
    assert a.distance_to(b) == approx(np.sqrt(8))


def test_gap():
    a = Wire(-1, 0, 0.5)
    b = Wire(1, 0, 0.5)
    assert a.gap_to(b) == approx(1)


def test_gap_two_radii():
    a = Wire(-1, 0, 0.5)
    b = Wire(1, 0, 1)
    assert a.gap_to(b) == approx(0.5)


def test_angle_to():
    a = Wire(-1, 0, 0.5)
    b = Wire(1, 0, 0.5)
    assert a.angle_to(b) == approx(-np.pi)
    assert b.angle_to(a) == approx(np.pi)


def test_angle_to_ortho():
    a = Wire(1, 0, 0.5)
    b = Wire(0, 1, 0.5)
    assert a.angle_to(b) == approx(0.5 * np.pi)


def test_offset():
    a = Wire(1, 1, 0.5)
    assert a.offset() == approx(np.sqrt(2))


def test_offset_zero():
    a = Wire(1, 2, 0.5)
    assert a.offset(1, 2) == approx(0)


def test_plane_height():
    a = Wire(0, 2.0, 0.5)
    p = Plane()
    assert p.height_of(a) == approx(2.0)


def test_shield_radius_zero():
    with pytest.raises(ValueError):
        Shield(0)


def test_shield_contains():
    a = Wire(0, 0.0, 0.5)
    s = Shield(1.0)
    assert s.contains(a)


def test_shield_contains_wire_larger():
    a = Wire(0, 0, 1.5)
    s = Shield(1.0)
    assert not s.contains(a)


def test_shield_contains_wire_on_edge():
    a = Wire(0, 0.5, 0.5)
    s = Shield(1.0)
    assert s.contains(a)


def test_shield_contains_outside():
    a = Wire(2.0, 0, 0.5)
    s = Shield(1.0)
    assert not s.contains(a)


def test_shield_contains_overlap():
    a = Wire(0.6, 0, 0.5)
    s = Shield(1.0)
    assert not s.contains(a)
