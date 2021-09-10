#!#!/bin/usr/python3

import pytest
from pytest import approx
import emtoolbox.geometry.geometry1d as gm


@pytest.mark.parametrize(
    "x, result",
    [
        (0.5, True),
        (0.4999, False),
        (2.5, True),
        (2.5001, False),
    ],
)
def test_rect_hit(x, result):
    rect = gm.Rect(0.5, 2.0)
    assert rect.hit(x) == result


def test_geometry_rect():
    rect = gm.Rect(0.5, 2.0)
    geom = gm.Geometry()
    geom.add_child(rect)
    assert geom.hit(1.0)
    assert not geom.hit(0.4)
    assert not geom.hit(2.6)


def test_geometry_negative_rect():
    rectp = gm.Rect(0.5, 2.0)  # Background 0.5 to 2.5
    rectn = gm.Rect(1.0, 1.0)  # Hole from 1 to 2
    rectn.positive = False
    geom = gm.Geometry()
    geom.add_child(rectp)
    geom.add_child(rectn)
    assert geom.hit(0.8)
    assert geom.hit(2.2)
    assert not geom.hit(0.4)
    assert not geom.hit(2.6)
    assert not geom.hit(1.5)


def test_geometry_rect_gaps():
    # Gaps from 1 to 1.3 and 1.7 to 2
    rectp = gm.Rect(0.5, 2.0)  # Background 0.5 to 2.5
    rectn = gm.Rect(1.0, 1.0)  # Hole from 1 to 2
    rectp2 = gm.Rect(1.3, 0.4)  # Rect from 1.3 to 1.7
    rectn.positive = False
    geom = gm.Geometry()
    geom.add_child(rectp)
    geom.add_child(rectn)
    geom.add_child(rectp2)
    assert geom.hit(0.8)
    assert geom.hit(2.2)
    assert not geom.hit(0.4)
    assert not geom.hit(2.6)
    assert geom.hit(1.5)
    assert not geom.hit(1.2)
    assert not geom.hit(1.8)


def test_geometry_bounds():
    rect = gm.Rect(0.5, 2.0)  # Background 0.5 to 2.5
    geom = gm.Geometry()
    geom.add_child(rect)
    assert geom.bounds() == (0.5, 2.5)


def test_geometry_bounds_two():
    recta = gm.Rect(0.5, 2.0)  # Background 0.5 to 2.5
    rectb = gm.Rect(-1.0, 12.0)  # Background 0.5 to 2.5
    geom = gm.Geometry()
    geom.add_child(recta)
    geom.add_child(rectb)
    assert geom.bounds() == (-1.0, 11.0)


def test_geometry_bounds_null():
    geom = gm.Geometry()
    with pytest.raises(Exception):
        geom.bounds()


def test_geometry_grid():
    rect = gm.Rect(0.0, 2.0)
    geom = gm.Geometry()
    geom.add_child(rect)
    assert geom.grid(5) == approx([0.0, 0.5, 1.0, 1.5, 2.0])


def test_geometry_mask():
    rect = gm.Rect(0.0, 2.0)
    rectn = gm.Rect(1.0, 0.5)
    rectn.positive = False
    geom = gm.Geometry()
    geom.add_child(rect)
    geom.add_child(rectn)
    grid = geom.grid(5)
    assert geom.mask(grid) == approx([1, 1, 0, 1])


def test_geometry_select():
    recta = gm.Rect(0.0, 4.0)
    rectb = gm.Rect(1.0, 2.0)
    rectc = gm.Rect(1.5, 1.0)
    recta.params['er'] = 2.0
    rectb.params['er'] = 3.0
    rectc.params['er'] = 4.0
    geom = gm.Geometry()
    geom.add_child(recta)
    geom.add_child(rectb)
    geom.add_child(rectc)
    grid = geom.grid(6)
    assert geom.select('er', grid) == approx([2, 3, 4, 3, 2])
