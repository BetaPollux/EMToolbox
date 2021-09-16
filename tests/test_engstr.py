#!/usr/bin/python3

import pytest
from pytest import approx
from emtoolbox.utils.conversions import engfloat, engstr


@pytest.mark.parametrize(
    "num, prec, result",
    [
        (300e6, 3, '300.000M'),
        (1e3, 1, '1.0k'),
        (5e-3, 2, '5.00m'),
        (0, 5, '0.00000'),
        (7e-6, 0, '7u'),
        (-1.2345, 2, '-1.23'),
        (-0.01e-3, 4, '-10.0000u'),
        (10000e15, 1, '1.0e+19'),
        (0.01e-15, 3, '1.000e-17'),
        (0.01e-15, 1, '1.0e-17')
    ],
)
def test_engstr(num, prec, result):
    assert engstr(num, precision=prec) == result


@pytest.mark.parametrize(
    "num, result",
    [
        ('300.000M', 3e8),
        ('1.0k', 1e3),
        ('5.00m', 5e-3),
        ('0.00000', 0.0),
        ('7u', 7e-6),
        ('-1.23', -1.23),
        ('-10.0000u', -10e-6),
        ('1.0e+19', 1e19),
        ('1.000e-17', 1e-17),
        ('3.2 k', 3.2e3),
        ('7.9e+2 m', 0.79),
        ('.55k', 550),
        ('+72M', 72e6),
        ('-.741 m', -741e-6)
    ],
)
def test_engfloat(num, result):
    assert engfloat(num) == approx(result)
