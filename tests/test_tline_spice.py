#!/usr/bin/python3

'''
LC:
One-segment
101----L1----201
    C1   C2
100----------200

Two-segment
101----L1--N0001--L2----201
    C1      C2       C3
100----------------------200
'''

from numpy import pi
from emtoolbox.tline.tline_spice import pi_model_2c
import pytest
from pytest import approx


@pytest.fixture
def lc_params():
    return {'L': 1e-6, 'C': 100e-12}


def get_values_of(name: str, lines: list):
    return [float(line.split(' ')[3])
            for line in lines if line.startswith(name)]


def test_get_values():
    lines = ['L11_1 1 2 1e-6', 'L11_2 3 4 5e-6', 'L11_3 5 6 10e-9']
    all = get_values_of('L11', lines)
    one = get_values_of('L11_2', lines)
    assert len(all) == 3
    assert len(one) == 1
    assert all[0] == 1e-6
    assert all[1] == 5e-6
    assert all[2] == 10e-9
    assert one[0] == 5e-6


def test_subckt_statement(lc_params):
    name = 'tline1'
    result = pi_model_2c(1, name=name, **lc_params)
    lines = result.split('\n')
    assert lines[0] == f'.SUBCKT {name} 100 101 201'
    assert lines[-2] == f'.ENDS {name}'
    assert lines[-1] == ''  # Should end with new line


@pytest.mark.parametrize(
    "N, length",
    [(1, 1.0), (3, 1.0), (100, 1.0),
    (3, 0.2), (3, 4.0), (3, 20.0)]
)
def test_inductance(lc_params, N, length):
    result = pi_model_2c(N, **lc_params, length=length)
    lines = result.split('\n')
    inductances = get_values_of('L11', lines)
    assert len(inductances) == N
    assert sum(inductances) == approx(lc_params['L'] * length, rel=0.001)
    assert len(set(inductances)) == 1


@pytest.mark.parametrize(
    "N, length",
    [(1, 1.0), (3, 1.0), (100, 1.0),
    (3, 0.2), (3, 4.0), (3, 20.0)]
)
def test_capacitance(lc_params, N, length):
    result = pi_model_2c(N, **lc_params, length=length)
    lines = result.split('\n')
    capacitances = get_values_of('C11', lines)
    assert len(capacitances) == N + 1
    assert sum(capacitances) == approx(lc_params['C'] * length, rel=0.001)
    if N > 1:
        assert len(set(capacitances)) == 2
        assert capacitances[0] == approx(0.5 * capacitances[1], rel=0.001)
