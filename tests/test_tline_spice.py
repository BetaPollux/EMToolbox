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

from emtoolbox.tline.tline_spice import pi_model_2c, pi_model_mtl, get_inductances, get_capacitances
import numpy as np
import pytest
from pytest import approx


@pytest.fixture
def tline_params():
    return {'L': 1e-6, 'C': 100e-12, 'R': 10, 'G': 1e-9}


@pytest.fixture
def mtl3c_params():
    return {'L': np.array([[5e-7, 1e-7],
                           [1e-7, 8e-7]]),
            'C': np.array([[5e-11, -2e-11],
                           [-2e-11, 7e-11]])
    }


@pytest.fixture
def mtl4c_params():
    return {'L': np.array([[5e-7, 1e-7, 2e-7],
                           [1e-7, 8e-7, 3e-7],
                           [2e-7, 3e-7, 6e-7]]),
            'C': np.array([[5e-11, -2e-11, -3e-11],
                           [-2e-11, 7e-11, -1e-11],
                           [-3e-11, -1e-11, 9e-11]])
    }


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


def test_subckt_statement(tline_params):
    name = 'tline1'
    result = pi_model_2c(1, name=name, **tline_params)
    lines = result.split('\n')
    assert f'.SUBCKT {name} 100 101 201' in lines
    assert lines[-2] == f'.ENDS {name}'
    assert lines[-1] == ''  # Should end with new line


@pytest.mark.parametrize(
    "R, G",
    [(0, 0), (3, 0), (0, 1e-9), (3, 1e-9)]
)
def test_inductor_statements(R, G):
    result = pi_model_2c(3, L=3e-6, C=120e-12, R=R, G=G)
    lines = result.split('\n')
    if R == 0:
        nets = ['N00001', 'N00001', 'N00002', 'N00002', '201']
    else:
        nets = ['N00001', 'N00002', 'N00003', 'N00004', 'N00005']
    assert f'L11_001 101 {nets[0]} 1.00000e-06' in lines
    assert f'L11_002 {nets[1]} {nets[2]} 1.00000e-06' in lines
    assert f'L11_003 {nets[3]} {nets[4]} 1.00000e-06' in lines


@pytest.mark.parametrize(
    "R, G",
    [(0, 0), (3, 0), (0, 1e-9), (3, 1e-9)]
)
def test_capacitor_statements(R, G):
    result = pi_model_2c(3, L=3e-6, C=120e-12, R=R, G=G)
    lines = result.split('\n')
    if R == 0:
        nets = ['N00001', 'N00002']
    else:
        nets = ['N00002', 'N00004']
    assert 'C11_001 101 100 2.00000e-11' in lines
    assert f'C11_002 {nets[0]} 100 4.00000e-11' in lines
    assert f'C11_003 {nets[1]} 100 4.00000e-11' in lines
    assert 'C11_004 201 100 2.00000e-11' in lines


@pytest.mark.parametrize(
    "G",
    [0, 1e-9]
)
def test_resistor_statements(G):
    result = pi_model_2c(3, L=3e-6, C=120e-12, R=3, G=G)
    lines = result.split('\n')
    assert 'R1_001 N00001 N00002 1.00000e+00' in lines
    assert 'R1_002 N00003 N00004 1.00000e+00' in lines
    assert 'R1_003 N00005 201 1.00000e+00' in lines


@pytest.mark.parametrize(
    "R",
    [0, 3]
)
def test_conductance_statements(R):
    result = pi_model_2c(3, L=3e-6, C=120e-12, R=R, G=1e-9)
    lines = result.split('\n')
    if R == 0:
        nets = ['N00001', 'N00002']
    else:
        nets = ['N00002', 'N00004']
    assert 'RG11_001 101 100 6.00000e+09' in lines
    assert f'RG11_002 {nets[0]} 100 3.00000e+09' in lines
    assert f'RG11_003 {nets[1]} 100 3.00000e+09' in lines
    assert 'RG11_004 201 100 6.00000e+09' in lines


@pytest.mark.parametrize(
    "N, length",
    [(1, 1.0), (3, 1.0), (100, 1.0),
    (3, 0.2), (3, 4.0), (3, 20.0)]
)
def test_inductance(tline_params, N, length):
    result = pi_model_2c(N, **tline_params, length=length)
    lines = result.split('\n')
    inductances = get_values_of('L11', lines)
    assert len(inductances) == N
    assert sum(inductances) == approx(tline_params['L'] * length, rel=0.001)
    assert len(set(inductances)) == 1


@pytest.mark.parametrize(
    "N, length",
    [(1, 1.0), (3, 1.0), (100, 1.0),
    (3, 0.2), (3, 4.0), (3, 20.0)]
)
def test_capacitance(tline_params, N, length):
    result = pi_model_2c(N, **tline_params, length=length)
    lines = result.split('\n')
    capacitances = get_values_of('C11', lines)
    assert len(capacitances) == N + 1
    assert sum(capacitances) == approx(tline_params['C'] * length, rel=0.001)
    if N > 1:
        assert len(set(capacitances)) == 2
        assert capacitances[0] == approx(0.5 * capacitances[1], rel=0.001)


@pytest.mark.parametrize(
    "N, length",
    [(1, 1.0), (3, 1.0), (100, 1.0),
    (3, 0.2), (3, 4.0), (3, 20.0)]
)
def test_resistance(tline_params, N, length):
    result = pi_model_2c(N, **tline_params, length=length)
    lines = result.split('\n')
    resistances = get_values_of('R1', lines)
    assert len(resistances) == N
    assert sum(resistances) == approx(tline_params['R'] * length, rel=0.001)
    assert len(set(resistances)) == 1


@pytest.mark.parametrize(
    "N, length",
    [(1, 1.0), (3, 1.0), (100, 1.0),
    (3, 0.2), (3, 4.0), (3, 20.0)]
)
def test_conductance(tline_params, N, length):
    result = pi_model_2c(N, **tline_params, length=length)
    lines = result.split('\n')
    resistances = get_values_of('RG11', lines)
    assert len(resistances) == N + 1
    assert sum([1 / r for r in resistances]) == approx(tline_params['G'] * length, rel=0.001)
    if N > 1:
        assert len(set(resistances)) == 2
        assert resistances[0] == approx(2 * resistances[1], rel=0.001)


def test_get_inductances():
    L = np.array([[1.11170e-6, 6.93901e-07],
                  [6.93901e-07, 1.38780e-06]])
    Le, Ke = get_inductances(L, length=2.54e-01)
    assert Le == approx(np.array([2.82372e-07, 3.52501e-07]))
    assert Ke == approx(np.array([[1.0, 5.585651e-01],
                                  [5.585651e-01, 1.0]]), rel=0.001)


def test_get_capacitances():
    C = np.array([[4.03439E-11, -2.01719E-11],
                  [-2.01719E-11, 2.95910E-11]])
    Ce = get_capacitances(C, length=2.54e-01)
    assert Ce == approx(np.array([[5.12368e-12, 5.12366e-12],
                                  [5.12366e-12, 2.39246e-12]]), rel=0.001)


def test_3c_inductor_statements(mtl3c_params):
    result = pi_model_mtl(1, **mtl3c_params)
    lines = result.split('\n')
    Le, Ke = get_inductances(mtl3c_params['L'])
    assert f'L11_001 101 201 {Le[0]:.5e}' in lines
    assert f'L22_001 102 202 {Le[1]:.5e}' in lines
    assert f'K12_001 L11_001 L22_001 {Ke[0, 1]:.5e}' in lines


def test_3c_capacitor_statements(mtl3c_params):
    result = pi_model_mtl(1, **mtl3c_params)
    lines = result.split('\n')
    Ce = get_capacitances(mtl3c_params['C'])
    assert f'C11_001 101 100 {0.5 * Ce[0, 0]:.5e}' in lines
    assert f'C11_002 201 100 {0.5 * Ce[0, 0]:.5e}' in lines
    assert f'C22_001 102 100 {0.5 * Ce[1, 1]:.5e}' in lines
    assert f'C22_002 202 100 {0.5 * Ce[1, 1]:.5e}' in lines
    assert f'C12_001 101 102 {0.5 * Ce[0, 1]:.5e}' in lines
    assert f'C12_002 201 202 {0.5 * Ce[0, 1]:.5e}' in lines


def test_4c_inductor_statements(mtl4c_params):
    result = pi_model_mtl(1, **mtl4c_params)
    lines = result.split('\n')
    Le, _ = get_inductances(mtl4c_params['L'])
    assert f'L11_001 101 201 {Le[0]:.5e}' in lines
    assert f'L22_001 102 202 {Le[1]:.5e}' in lines
    assert f'L33_001 103 203 {Le[2]:.5e}' in lines


def test_4c_coupling_statements(mtl4c_params):
    result = pi_model_mtl(1, **mtl4c_params)
    lines = result.split('\n')
    _, Ke = get_inductances(mtl4c_params['L'])
    assert f'K12_001 L11_001 L22_001 {Ke[0, 1]:.5e}' in lines
    assert f'K13_001 L11_001 L33_001 {Ke[0, 2]:.5e}' in lines
    assert f'K23_001 L22_001 L33_001 {Ke[1, 2]:.5e}' in lines


def test_4c_self_capacitor_statements(mtl4c_params):
    result = pi_model_mtl(1, **mtl4c_params)
    lines = result.split('\n')
    Ce = get_capacitances(mtl4c_params['C'])
    assert f'C11_001 101 100 {0.5 * Ce[0, 0]:.5e}' in lines
    assert f'C11_002 201 100 {0.5 * Ce[0, 0]:.5e}' in lines
    assert f'C22_001 102 100 {0.5 * Ce[1, 1]:.5e}' in lines
    assert f'C22_002 202 100 {0.5 * Ce[1, 1]:.5e}' in lines
    assert f'C33_001 103 100 {0.5 * Ce[2, 2]:.5e}' in lines
    assert f'C33_002 203 100 {0.5 * Ce[2, 2]:.5e}' in lines


def test_4c_mutual_capacitor_statements(mtl4c_params):
    result = pi_model_mtl(1, **mtl4c_params)
    lines = result.split('\n')
    Ce = get_capacitances(mtl4c_params['C'])
    assert f'C12_001 101 102 {0.5 * Ce[0, 1]:.5e}' in lines
    assert f'C12_002 201 202 {0.5 * Ce[0, 1]:.5e}' in lines
    assert f'C13_001 101 103 {0.5 * Ce[0, 2]:.5e}' in lines
    assert f'C13_002 201 203 {0.5 * Ce[0, 2]:.5e}' in lines
    assert f'C23_001 102 103 {0.5 * Ce[1, 2]:.5e}' in lines
    assert f'C23_002 202 203 {0.5 * Ce[1, 2]:.5e}' in lines


def test_4c_inductance(mtl4c_params):
    N = 1
    result = pi_model_mtl(N, **mtl4c_params)
    lines = result.split('\n')
    Le, _ = get_inductances(mtl4c_params['L'])
    for i in range(3):
        inductances = get_values_of(f'L{i+1}{i+1}', lines)
        assert len(inductances) == N
        assert sum(inductances) == approx(Le[i], rel=0.001)
        assert len(set(inductances)) == 1


def test_4c_coupling(mtl4c_params):
    N = 1
    result = pi_model_mtl(N, **mtl4c_params)
    lines = result.split('\n')
    _, Ke = get_inductances(mtl4c_params['L'])
    for i in range(3):
        for j in range(i + 1, 3):
            couplings = get_values_of(f'K{i+1}{j+1}', lines)
            assert len(couplings) == N
            assert sum(couplings) == approx(Ke[i, j], rel=0.001)
            assert len(set(couplings)) == 1


def test_4c_capacitance(mtl4c_params):
    N = 1
    result = pi_model_mtl(N, **mtl4c_params)
    lines = result.split('\n')
    Ce = get_capacitances(mtl4c_params['C'])
    for i in range(3):
        capacitances = get_values_of(f'C{i+1}{i+1}', lines)
        assert len(capacitances) == N + 1
        assert sum(capacitances) == approx(Ce[i, i], rel=0.001)
        assert len(set(capacitances)) == 1


def test_4c_mutual_capacitance(mtl4c_params):
    N = 1
    result = pi_model_mtl(N, **mtl4c_params)
    lines = result.split('\n')
    Ce = get_capacitances(mtl4c_params['C'])
    for i in range(3):
        for j in range(i + 1, 3):
            capacitances = get_values_of(f'C{i+1}{j+1}', lines)
            assert len(capacitances) == N + 1
            assert sum(capacitances) == approx(Ce[i, j], rel=0.001)
            assert len(set(capacitances)) == 1
