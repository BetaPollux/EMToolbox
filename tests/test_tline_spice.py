#!/usr/bin/python3

'''
LC:
One-segment
02----L1----03
    C1  C2
01----------01

Two-segment
02----L1--03--L2----04
    C1    C2    C3
01------------------01
'''

from emtoolbox.tline.tline_spice import get_pi_model
import pytest
from pytest import approx

def get_values_of(name: str, lines: list):
    return [float(line.split(' ')[3]) for line in lines if line.startswith(name)]

def test_get_values():
    lines = ['YAY 1 2 1e-6', 'NAY 3 4 5e-6', 'YAY 5 6 10e-7']
    yay = get_values_of('YAY', lines)
    nay = get_values_of('NAY', lines)
    nope = get_values_of('NOPE', lines)
    assert len(yay) == 2
    assert len(nay) == 1
    assert len(nope) == 0
    assert yay[0] == 1e-6
    assert yay[1] == 10e-7
    assert nay[0] == 5e-6

def test_checks_for_inductance():
    rlgc = { 'c': 4e-12 }
    with pytest.raises(Exception):
        get_pi_model('tline', rlgc, 1.0, 1)

def test_checks_for_capacitance():
    rlgc = { 'l': 1e-6 }
    with pytest.raises(Exception):
        get_pi_model('tline', rlgc, 1.0, 1)

def test_lc_subckt_1seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12 }
    result = get_pi_model('tline1', rlgc, 1.0, 1)
    lines = result.split('\n')
    assert '.SUBCKT tline1 N00001 N00002 N00003' in lines
    assert '.ENDS tline1' in lines

def test_lc_1seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12 }
    result = get_pi_model('tline', rlgc, 1.0, 1)
    lines = result.split('\n')
    assert 'LS001 N00002 N00003 1.00000e-06' in lines
    assert 'CP001 N00002 N00001 2.00000e-12' in lines
    assert 'CP002 N00003 N00001 2.00000e-12' in lines

def test_lc_1seg_length():
    rlgc = { 'l': 1e-6, 'c': 4e-12 }
    result = get_pi_model('tline', rlgc, 2.0, 1)
    lines = result.split('\n')
    assert 'LS001 N00002 N00003 2.00000e-06' in lines
    assert 'CP001 N00002 N00001 4.00000e-12' in lines
    assert 'CP002 N00003 N00001 4.00000e-12' in lines

def test_lc_subckt_2seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12 }
    result = get_pi_model('tline2', rlgc, 1.0, 2)
    lines = result.split('\n')
    assert '.SUBCKT tline2 N00001 N00002 N00004' in lines
    assert '.ENDS tline2' in lines

def test_lc_2seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12 }
    result = get_pi_model('tline', rlgc, 1.0, 2)
    lines = result.split('\n')
    assert 'LS001 N00002 N00003 5.00000e-07' in lines
    assert 'CP001 N00002 N00001 1.00000e-12' in lines
    assert 'CP002 N00003 N00001 2.00000e-12' in lines
    assert 'LS002 N00003 N00004 5.00000e-07' in lines
    assert 'CP003 N00004 N00001 1.00000e-12' in lines

def test_rlc_subckt_1seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'r': 1e-3 }
    result = get_pi_model('tline1', rlgc, 1.0, 1)
    lines = result.split('\n')
    assert '.SUBCKT tline1 N00001 N00002 N00004' in lines
    assert '.ENDS tline1' in lines

def test_rlc_1seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'r': 1e-3 }
    result = get_pi_model('tline', rlgc, 1.0, 1)
    lines = result.split('\n')
    assert 'LS001 N00002 N00003 1.00000e-06' in lines
    assert 'CP001 N00002 N00001 2.00000e-12' in lines
    assert 'CP002 N00004 N00001 2.00000e-12' in lines
    assert 'RS001 N00003 N00004 1.00000e-03' in lines

def test_rlc_subckt_2seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'r': 1e-3 }
    result = get_pi_model('tline2', rlgc, 1.0, 2)
    lines = result.split('\n')
    assert '.SUBCKT tline2 N00001 N00002 N00006' in lines
    assert '.ENDS tline2' in lines

def test_rlc_2seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'r': 1e-3 }
    result = get_pi_model('tline', rlgc, 1.0, 2)
    lines = result.split('\n')
    assert 'LS001 N00002 N00003 5.00000e-07' in lines
    assert 'CP001 N00002 N00001 1.00000e-12' in lines
    assert 'CP002 N00004 N00001 2.00000e-12' in lines
    assert 'LS002 N00004 N00005 5.00000e-07' in lines
    assert 'CP003 N00006 N00001 1.00000e-12' in lines
    assert 'RS001 N00003 N00004 5.00000e-04' in lines
    assert 'RS002 N00005 N00006 5.00000e-04' in lines

def test_lgc_subckt_1seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'g': 1e-9 }
    result = get_pi_model('tline1', rlgc, 1.0, 1)
    lines = result.split('\n')
    assert '.SUBCKT tline1 N00001 N00002 N00003' in lines
    assert '.ENDS tline1' in lines

def test_lgc_1seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'g': 1e-9 }
    result = get_pi_model('tline', rlgc, 1.0, 1)
    lines = result.split('\n')
    assert 'LS001 N00002 N00003 1.00000e-06' in lines
    assert 'CP001 N00002 N00001 2.00000e-12' in lines
    assert 'CP002 N00003 N00001 2.00000e-12' in lines
    assert 'RP001 N00002 N00001 2.00000e+09' in lines
    assert 'RP002 N00003 N00001 2.00000e+09' in lines

def test_lgc_2seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'g': 1e-9 }
    result = get_pi_model('tline', rlgc, 1.0, 2)
    lines = result.split('\n')
    assert 'LS001 N00002 N00003 5.00000e-07' in lines
    assert 'LS002 N00003 N00004 5.00000e-07' in lines
    assert 'CP001 N00002 N00001 1.00000e-12' in lines
    assert 'CP002 N00003 N00001 2.00000e-12' in lines
    assert 'CP003 N00004 N00001 1.00000e-12' in lines
    assert 'RP001 N00002 N00001 4.00000e+09' in lines
    assert 'RP002 N00003 N00001 2.00000e+09' in lines
    assert 'RP003 N00004 N00001 4.00000e+09' in lines

def test_inductance_100seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'r': 1e-3, 'g': 1e-9 }
    result = get_pi_model('tline', rlgc, 1.0, 100)
    lines = result.split('\n')
    inductances = get_values_of('LS', lines)
    assert len(inductances) == 100
    assert sum(inductances) == approx(rlgc['l'])
    assert len(set(inductances)) == 1

def test_capacitance_100seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'r': 1e-3, 'g': 1e-9 }
    result = get_pi_model('tline', rlgc, 1.0, 100)
    lines = result.split('\n')
    capacitances = get_values_of('CP', lines)
    assert len(capacitances) == 101
    assert capacitances[0] == approx(0.5 * capacitances[1])
    assert capacitances[-1] == approx(0.5 * capacitances[1])
    assert sum(capacitances) == approx(rlgc['c'])
    assert len(set(capacitances)) == 2

def test_resistance_100seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'r': 1e-3, 'g': 1e-9 }
    result = get_pi_model('tline', rlgc, 1.0, 100)
    lines = result.split('\n')
    resistances = get_values_of('RS', lines)
    assert len(resistances) == 100
    assert sum(resistances) == approx(rlgc['r'])
    assert len(set(resistances)) == 1

def test_conductance_100seg():
    rlgc = { 'l': 1e-6, 'c': 4e-12, 'r': 1e-3, 'g': 1e-9 }
    result = get_pi_model('tline', rlgc, 1.0, 100)
    lines = result.split('\n')
    resistances = get_values_of('RP', lines)
    assert len(resistances) == 101
    assert resistances[0] == approx(2 * resistances[1])
    assert resistances[-1] == approx(2 * resistances[1])
    assert sum([1 / r for r in resistances]) == approx(rlgc['g'])
    assert len(set(resistances)) == 2
