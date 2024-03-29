#!/usr/bin/python3

import numpy as np
from emtoolbox.fields.pplate import ParallelPlates
from emtoolbox.fields.coaxcap import CoaxCapacitor
from emtoolbox.fields.spherecap import SphereCapacitor
import emtoolbox.fields.poisson_fdm as fdm
import pytest
from pytest import approx
import matplotlib.pyplot as plt


def test_trough_analytical():
    w = 3.0
    h = 4.0
    x = np.linspace(0, w, 4)
    y = np.linspace(0, h, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    V = fdm.trough_analytical(X, Y, v_left=10, v_bottom=0, v_top=20, v_right=30)
    assert V[1, 3] == approx(16.4784, abs=1e-4)
    assert V[2, 3] == approx(21.8499, abs=1e-4)
    assert V[1, 2] == approx(14.1575, abs=1e-4)
    assert V[2, 2] == approx(20.4924, abs=1e-4)
    assert V[1, 1] == approx(9.60942, abs=1e-4)
    assert V[2, 1] == approx(14.9810, abs=1e-4)


def test_poisson_1d():
    pp = ParallelPlates(1.0, 5e-3)
    X, _ = pp.get_arrays(101)
    v0 = -2
    v1 = 5
    V = fdm.poisson_1d(X, v_left=v0, v_right=v1, conv=1e-5)
    Va = pp.potential(X, v1 - v0, v0)
    assert V == approx(Va, abs=0.01)


def test_poisson_1d_dielectric():
    pp = ParallelPlates((5.0, 1.0), (1e-3, 3e-3))
    v0 = 0
    v1 = 200
    X, er = pp.get_arrays(101)
    V = fdm.poisson_1d(X, dielectric=er, v_left=v0, v_right=v1, conv=1e-7)
    Va = pp.potential(X, v1 - v0, v0)
    assert V == approx(Va, abs=0.01)


def test_poisson_1d_dielectric2():
    pp = ParallelPlates((1.0, 5.0, 1.0), (2e-3, 1e-3, 1e-3))
    v0 = 0
    v1 = 200
    X, er = pp.get_arrays(101)
    V = fdm.poisson_1d(X, dielectric=er, v_left=v0, v_right=v1, conv=1e-7)
    Va = pp.potential(X, v1 - v0, v0)
    assert V == approx(Va, abs=0.01)


def test_poisson_1d_unity_dielectric():
    X = np.linspace(0, 5, 11)
    er = np.ones_like(X[1:])
    v0 = 10
    V = fdm.poisson_1d(X, v_left=v0, conv=1e-7)
    Ver = fdm.poisson_1d(X, dielectric=er, v_left=v0, conv=1e-7)
    assert V == approx(Ver)


def test_poisson_2d():
    w = 2.0
    h = 1.0
    x = np.linspace(0, w, 101)
    y = np.linspace(0, h, 51)
    X, Y = np.meshgrid(x, y, indexing='ij')
    bc = {'v_top': 10, 'v_left': 5, 'v_right': -2, 'v_bottom': -4}
    V = fdm.poisson_2d(X, Y, **bc, conv=1e-5)
    Va = fdm.trough_analytical(X, Y, **bc)
    # Exclude boundaries due to analytical error at corners
    assert V[1:-1, 1:-1] == approx(Va[1:-1, 1:-1], abs=0.1)


def test_poisson_2d_unity_dielectric():
    x = np.linspace(0, 5, 11)
    y = np.linspace(0, 5, 11)
    X, Y = np.meshgrid(x, y, indexing='ij')
    er = np.ones_like(X[1:, 1:])
    v0 = 10
    V = fdm.poisson_2d(X, Y, v_left=v0, conv=1e-7)
    Ver = fdm.poisson_2d(X, Y, dielectric=er, v_left=v0, conv=1e-7)
    assert V == approx(Ver)


def test_poisson_2d_indexing():
    x = np.linspace(0, 2.0, 21)
    y = np.linspace(0, 1.0, 11)
    X, Y = np.meshgrid(x, y, indexing='ij')
    with pytest.raises(Exception):
        fdm.poisson_2d(X, Y)


def test_poisson_2d_spacing():
    x = np.linspace(0, 2.0, 21)
    y = np.linspace(0, 2.0, 11)
    X, Y = np.meshgrid(x, y)
    with pytest.raises(Exception):
        fdm.poisson_2d(X, Y)


def test_gauss_1d():
    pp = ParallelPlates(2.0, 15e-3)
    X, er = pp.get_arrays(101)
    v1 = 10.0
    V = pp.potential(X, v1)
    Q = np.array([fdm.gauss_1d(X, V, er, i) for i in range(1, len(X) - 1)])
    Qa = pp.charge(v1)
    assert Q == approx(Qa)


def test_gauss_1d_dielectric():
    pp = ParallelPlates((2.0, 4.0), (3e-3, 2e-3))
    X, er = pp.get_arrays(101)
    v1 = 10.0
    V = pp.potential(X, v1)
    Q = np.array([fdm.gauss_1d(X, V, er, i) for i in range(1, len(X) - 1)])
    Qa = pp.charge(v1)
    assert Q == approx(Qa)


def test_gauss_1d_coax():
    cc = CoaxCapacitor(0.5e-3, 5.2, 3.5e-3)
    X, er = cc.get_arrays(101)
    v1 = 10.0
    V = cc.potential(X, Va=v1)
    # 1D gauss returns a charge density over circumference
    # Result is negative due to direction of X
    Q = np.array([-2*np.pi*X[i]*fdm.gauss_1d(X, V, er, i) for i in range(1, len(X) - 1)])
    Qa = cc.charge(v1)
    assert Q == approx(Qa, rel=0.01)


def test_poisson_1d_bc():
    X = np.linspace(0, 5, 5)
    v0 = -2.0
    v1 = 5.0
    v2 = 8.0
    bc_val = np.zeros_like(X)
    bc_val[2] = v1
    bc_bool = bc_val > 0
    bc = (bc_bool, bc_val)
    V = fdm.poisson_1d(X, v_left=v0, v_right=v2, bc=bc, conv=1e-3, sor=1)
    assert V == approx([-2, 1.5, 5, 6.5, 8])


def test_poisson_1d_bc_mult():
    X = np.linspace(0, 5, 7)
    v0 = -2.0
    v1 = 5.0
    v2 = 8.0
    v3 = 20.0
    bc_val = np.zeros_like(X)
    bc_val[2] = v1
    bc_val[4] = v2
    bc_bool = bc_val > 0
    bc = (bc_bool, bc_val)
    V = fdm.poisson_1d(X, v_left=v0, v_right=v3, bc=bc, conv=1e-3, sor=1)
    assert V == approx([-2, 1.5, 5, 6.5, 8, 14, 20])


def test_poisson_1d_bc_slice():
    X = np.linspace(0, 5, 7)
    v0 = -2.0
    v1 = 5.0
    v2 = 8.0
    bc_val = np.zeros_like(X)
    bc_val[2:5] = v1
    bc_bool = bc_val > 0
    bc = (bc_bool, bc_val)
    V = fdm.poisson_1d(X, v_left=v0, v_right=v2, bc=bc, conv=1e-3, sor=1)
    assert V == approx([-2, 1.5, 5, 5, 5, 6.5, 8])


def test_poisson_2d_bc():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    bc_val = np.zeros_like(X)
    bc_val[2, 2] = 12.0
    bc_bool = bc_val > 0
    bc = (bc_bool, bc_val)
    V = fdm.poisson_2d(X, Y, bc=bc, conv=1e-6, sor=1)
    assert V == approx(np.array([[0, 0, 0, 0, 0],
                                 [0, 2, 4, 2, 0],
                                 [0, 4, 12, 4, 0],
                                 [0, 2, 4, 2, 0],
                                 [0, 0, 0, 0, 0]]))


def test_poisson_2d_bc_mult():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    vbc1 = 13
    vbc2 = 7
    bc_val = np.zeros_like(X)
    bc_val[1, 3] = vbc1
    bc_val[3, 1] = vbc2
    bc_bool = bc_val > 0
    bc = (bc_bool, bc_val)
    V = fdm.poisson_2d(X, Y, bc=bc, conv=1e-6, sor=1)
    assert V[1, 3] == approx(vbc1)
    assert V[3, 1] == approx(vbc2)


def test_poisson_2d_bc_slice():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    vbc = 16
    bc_val = np.zeros_like(X)
    bc_val[1:4, 2] = vbc
    bc_bool = bc_val > 0
    bc = (bc_bool, bc_val)
    V = fdm.poisson_2d(X, Y, bc=bc, conv=1e-6, sor=1)
    assert V[1:4, 2] == approx(vbc)


def test_poisson_2d_coax():
    ri = 2.0e-3
    ro = 4.0e-3
    w = 1.1 * ro
    N = 101
    Va = 10.0
    x = np.linspace(-w, w, N)
    y = np.linspace(-w, w, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    cc = CoaxCapacitor(ri, 1.0, ro - ri)
    expected = cc.potential(X, Y, Va=Va)
    potential = fdm.poisson_2d(X, Y, bc=bc, conv=1e-5)
    assert potential == approx(expected, abs=0.4)


def test_poisson_2d_coax_2layer():
    ri = 2.0e-3
    re = 2.8e-3
    ro = 4.0e-3
    er1, er2 = 4.0, 1.0
    w = 1.1 * ro
    N = 101
    Va = 10.0
    x = np.linspace(-w, w, N)
    y = np.linspace(-w, w, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    er = np.select([R <= re, R > re], [er1, er2])[:-1, :-1]
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    cc = CoaxCapacitor(ri, (er1, er2), (re - ri, ro - re))
    expected = cc.potential(X, Y, Va=Va)
    np.savetxt('e2d.csv', expected, delimiter=',', fmt='%.3f')
    potential = fdm.poisson_2d(X, Y, dielectric=er, bc=bc, conv=1e-5)
    assert potential == approx(expected, abs=0.6)


def test_poisson_2d_dielectric():
    w = 2.0
    h = 1.0
    x = np.linspace(0, w, 41)
    y = np.linspace(0, h, 21)
    X, Y = np.meshgrid(x, y, indexing='ij')
    bc = {'v_top': 10, 'v_left': 5, 'v_right': -2, 'v_bottom': -4}
    er = np.ones_like(X)[:-1, :-1]
    V1 = fdm.poisson_2d(X, Y, **bc, conv=1e-3)
    V2 = fdm.poisson_2d(X, Y, **bc, dielectric=er, conv=1e-3)
    assert V1 == approx(V2)


def test_gauss_2d_coax():
    ri = 1.0e-3
    ro = 4.0e-3
    rm = 0.5 * (ri + ro)
    assert rm * 1.414 < ro
    w = 1.1 * ro
    dx = ri / 40
    Va = 10.0
    x = np.arange(0, w, dx)
    y = np.arange(-w, w, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    cc = CoaxCapacitor(ri, 1.0, ro - ri)
    V = cc.potential(X, Y, Va=Va)
    expected = cc.charge(Va)
    er = np.ones_like(V)[:-1, :-1]
    xi1 = np.searchsorted(x, -rm)
    xi2 = np.searchsorted(x, rm)
    yi1 = np.searchsorted(y, -rm)
    yi2 = np.searchsorted(y, rm)
    gauss = fdm.gauss_2d(X, Y, V, er, xi1, xi2, yi1, yi2)
    assert gauss == approx(expected, rel=0.01, abs=1e-14)


def test_poisson_2d_coax_xsym():
    ri = 2.0e-3
    ro = 4.0e-3
    w = 1.1 * ro
    dx = ri / 40
    Va = 10.0
    x = np.arange(0, w, dx)
    y = np.arange(-w, w, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    cc = CoaxCapacitor(ri, 1.0, ro - ri)
    expected = cc.potential(X, Y, Va=Va)
    potential = fdm.poisson_2d(X, Y, bc=bc, conv=1e-6, xsym=True)
    assert potential == approx(expected, abs=0.4)


def test_poisson_2d_coax_ysym():
    ri = 2.0e-3
    ro = 4.0e-3
    w = 1.1 * ro
    dx = ri / 40
    Va = 10.0
    x = np.arange(-w, w, dx)
    y = np.arange(0, w, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    cc = CoaxCapacitor(ri, 1.0, ro - ri)
    expected = cc.potential(X, Y, Va=Va)
    potential = fdm.poisson_2d(X, Y, bc=bc, conv=1e-6, ysym=True)
    assert potential == approx(expected, abs=0.4)


def test_poisson_2d_coax_xysym():
    ri = 2.0e-3
    ro = 4.0e-3
    w = 1.1 * ro
    dx = ri / 40
    Va = 10.0
    x = np.arange(0, w, dx)
    y = np.arange(0, w, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    cc = CoaxCapacitor(ri, 1.0, ro - ri)
    expected = cc.potential(X, Y, Va=Va)
    potential = fdm.poisson_2d(X, Y, bc=bc, conv=1e-6, xsym=True, ysym=True)
    assert potential == approx(expected, abs=0.4)


def test_gauss_2d_coax_xsym():
    ri = 1.0e-3
    ro = 4.0e-3
    rm = 0.5 * (ri + ro)
    assert rm * 1.414 < ro
    w = 1.1 * ro
    dx = ri / 40
    Va = 10.0
    x = np.arange(0, w, dx)
    y = np.arange(-w, w, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    cc = CoaxCapacitor(ri, 1.0, ro - ri)
    V = cc.potential(X, Y, Va=Va)
    expected = cc.charge(Va)
    er = np.ones_like(V)[:-1, :-1]
    xi1 = 0
    xi2 = np.searchsorted(x, rm)
    yi1 = np.searchsorted(y, -rm)
    yi2 = np.searchsorted(y, rm)
    gauss = fdm.gauss_2d(X, Y, V, er, xi1, xi2, yi1, yi2)
    assert gauss == approx(expected, rel=0.01, abs=1e-14)


def test_gauss_2d_coax_ysym():
    ri = 1.0e-3
    ro = 4.0e-3
    rm = 0.5 * (ri + ro)
    assert rm * 1.414 < ro
    w = 1.1 * ro
    dx = ri / 40
    Va = 10.0
    x = np.arange(-w, w, dx)
    y = np.arange(0, w, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    cc = CoaxCapacitor(ri, 1.0, ro - ri)
    V = cc.potential(X, Y, Va=Va)
    expected = cc.charge(Va)
    er = np.ones_like(V)[:-1, :-1]
    xi1 = np.searchsorted(x, -rm)
    xi2 = np.searchsorted(x, rm)
    yi1 = 0
    yi2 = np.searchsorted(y, rm)
    gauss = fdm.gauss_2d(X, Y, V, er, xi1, xi2, yi1, yi2)
    assert gauss == approx(expected, rel=0.01, abs=1e-14)


def test_gauss_2d_coax_xysym():
    ri = 1.0e-3
    ro = 4.0e-3
    rm = 0.5 * (ri + ro)
    assert rm * 1.414 < ro
    w = 1.1 * ro
    dx = ri / 40
    Va = 10.0
    x = np.arange(0, w, dx)
    y = np.arange(0, w, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    cc = CoaxCapacitor(ri, 1.0, ro - ri)
    V = cc.potential(X, Y, Va=Va)
    expected = cc.charge(Va)
    er = np.ones_like(V)[:-1, :-1]
    xi1 = 0
    xi2 = np.searchsorted(x, rm)
    yi1 = 0
    yi2 = np.searchsorted(y, rm)
    gauss = fdm.gauss_2d(X, Y, V, er, xi1, xi2, yi1, yi2)
    assert gauss == approx(expected, rel=0.01, abs=1e-14)


def test_poisson_3d_unity_dielectric():
    x = np.linspace(0, 5, 11)
    y = np.linspace(0, 5, 11)
    z = np.linspace(0, 5, 11)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    er = np.ones_like(X[1:, 1:, 1:])
    v0 = 10
    V = fdm.poisson_3d(X, Y, Z, v_left=v0, conv=1e-7)
    Ver = fdm.poisson_3d(X, Y, Z, dielectric=er, v_left=v0, conv=1e-7)
    assert V == approx(Ver)


def test_poisson_3d_sphere():
    ri = 2.0e-3
    ro = 4.0e-3
    w = 1.1 * ro
    N = 61
    dx = 2 * w / N
    Va = 10.0
    x = np.arange(-w, w, dx)
    y = np.arange(-w, w, dx)
    z = np.arange(-w, w, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    sc = SphereCapacitor(ri, 1.0, ro - ri)
    expected = sc.potential(X, Y, Z, Va=Va)
    potential = fdm.poisson_3d(X, Y, Z, bc=bc, conv=1e-5)
    assert potential == approx(expected, abs=0.8)


def test_poisson_3d_sphere_2layer():
    ri = 2.0e-3
    re = 2.8e-3
    ro = 4.0e-3
    er1, er2 = 4.0, 1.0
    w = 1.1 * ro
    N = 61
    dx = 2 * w / N
    Va = 10.0
    x = np.arange(-w, w, dx)
    y = np.arange(-w, w, dx)
    z = np.arange(-w, w, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    er = np.select([R <= re, R > re], [er1, er2])[:-1, :-1, :-1]
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    sc = SphereCapacitor(ri, (er1, er2), (re - ri, ro - re))
    expected = sc.potential(X, Y, Z, Va=Va)
    potential = fdm.poisson_3d(X, Y, Z, dielectric=er, bc=bc, conv=1e-7)
    assert potential == approx(expected, abs=0.8)


@pytest.mark.parametrize(
    "xsym, ysym, zsym",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),    # TODO xy sym fails
        (False, False, True),
        (True, False, True),    # TODO xz sym fails
        (False, True, True),    # TODO yz sym fails
        (True, True, True)      # TODO xyz sym fails
    ],
)
def test_poisson_3d_sphere_sym(xsym, ysym, zsym):
    ri = 2.0e-3
    ro = 4.0e-3
    w = 1.1 * ro
    N = 61
    dx = 2 * w / N
    Va = 10.0
    x = np.arange(-w * (not xsym), w, dx)
    y = np.arange(-w * (not ysym), w, dx)
    z = np.arange(-w * (not zsym), w, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    bc_bool = np.logical_or(R < ri, R > ro)
    bc_val = np.select([R < ri, R > ro], [Va, 0])
    bc = (bc_bool, bc_val)
    sc = SphereCapacitor(ri, 1.0, ro - ri)
    expected = sc.potential(X, Y, Z, Va=Va)
    potential = fdm.poisson_3d(X, Y, Z, bc=bc, conv=1e-5, xsym=xsym, ysym=ysym, zsym=zsym)
    assert potential == approx(expected, abs=0.9)
