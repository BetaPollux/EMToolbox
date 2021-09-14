#!#!/bin/usr/python3

import pytest
from pytest import approx
import emtoolbox.geometry.geometry2d as gm


@pytest.mark.parametrize(
    "x, y, result",
    [
        (0.5, 1.0, True),
        (0.4999, 1.0, False),
        (2.5, 1.0, True),
        (2.5001, 1.0, False),
        (0.5, 1.0, True),
        (0.5, 1.0001, False),
        (0.5, 0.5, True),
        (0.5, 0.4999, False),
    ],
)
def test_rect_hit(x, y, result):
    rect = gm.Rect(0.5, 2.0, 1.0, 0.5)
    assert rect.hit(x, y) == result


@pytest.mark.parametrize(
    "x, y, result",
    [
        (1.0, 1.0, True),
        (0.5, 1.0, True),
        (1.5, 1.0, True),
        (1.0, 0.5, True),
        (1.0, 1.5, True),
        (1.4, 1.4, False),
        (0.6, 0.6, False),
        (1.3, 1.3, True)
    ],
)
def test_circ_hit(x, y, result):
    circ = gm.Circ(1.0, 1.0, 0.5)
    assert circ.hit(x, y) == result


def test_geometry_rect():
    rect = gm.Rect(0.5, 2.0, 1.0, 0.5)
    geom = gm.Geometry()
    geom.add_child(rect)
    assert geom.hit(1.0, 0.75)
    assert not geom.hit(0.4, 0.75)
    assert not geom.hit(2.6, 0.75)
    assert not geom.hit(1.0, 1.1)
    assert not geom.hit(1.0, 0.4)
