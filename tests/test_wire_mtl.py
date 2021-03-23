import numpy as np
import pytest
from pytest import approx
import emtoolbox.tline.wire_mtl as mtl
from emtoolbox.tline.wire import Wire, Plane, Shield
import emtoolbox.tline.coax as coax


def test_wire_capacitance():
    # Paul MTL P4.2
    rw = 7.5
    s = 50
    C = mtl.wire_capacitance(s, rw)
    C2 = mtl.wire_capacitance(s, rw, rw)
    assert C == approx(14.8e-12, abs=0.1e-12)
    assert C == approx(C2, abs=0.1e-12)


def test_wire_capacitance_er():
    rw = 7.5
    s = 50
    er = 4
    C = mtl.wire_capacitance(s, rw)
    Cer = mtl.wire_capacitance(s, rw, er=er)
    assert C == approx(Cer / er, abs=1e-14)


def test_wire_inductance():
    # Paul MTL P4.2
    rw = 7.5
    s = 50
    L = mtl.wire_inductance(s, rw)
    L2 = mtl.wire_inductance(s, rw, rw)
    assert L == approx(749e-9, abs=1e-9)
    assert L == approx(L2, abs=1e-9)


def test_wire_capacitance_too_close():
    rw = 10
    s = 19
    with pytest.raises(Exception):
        mtl.wire_capacitance(s, rw)


def test_wire_capacitance_rw2_too_close():
    rw1 = 10
    rw2 = 5
    s = 14
    with pytest.raises(Exception):
        mtl.wire_capacitance(s, rw1, rw2)


def test_wire_inductance_too_close():
    rw = 10
    s = 19
    with pytest.raises(Exception):
        mtl.wire_inductance(s, rw)


def test_wire_inductance_rw2_too_close():
    rw1 = 10
    rw2 = 5
    s = 14
    with pytest.raises(Exception):
        mtl.wire_inductance(s, rw1, rw2)


def test_wire_self_inductance():
    # Paul MTL P5.4
    rw = 7.5
    di0 = 50
    ref = Wire(0, 0, rw)
    wire = Wire(di0, 0, rw)
    L = mtl.wire_self_inductance(wire, ref)
    assert L == approx(0.759e-6, abs=0.001e-6)


def test_wire_mutual_inductance():
    # Paul MTL P5.4
    rw = 7.5
    ref = Wire(0, 0, rw)
    wi = Wire(-50, 0, rw)
    wj = Wire(50, 0, rw)
    L = mtl.wire_mutual_inductance(wi, wj, ref)
    assert L == approx(0.241e-6, rel=0.001)


def test_bad_ref():
    with pytest.raises(TypeError):
        mtl.WireMtl([Wire(0, 0, 0.5)], 'Ref!')


def test_bad_wire():
    with pytest.raises(TypeError):
        mtl.WireMtl([Plane()], Plane())


def test_empty_list():
    with pytest.raises(ValueError):
        mtl.WireMtl([], Plane())


def test_two_wire_capacitance():
    rw = 7.5
    s = 50
    wires = [Wire(-0.5 * s, 0, rw)]
    ref = Wire(0.5 * s, 0, rw)
    pair = mtl.WireMtl(wires, ref)
    C = mtl.wire_capacitance(s, rw)
    assert pair.capacitance()[[0]] == approx(C, rel=0.02, abs=1e-14)


def test_two_wire_capacitance_diagonal():
    rw = 7.5
    s = 50
    x2 = y2 = s / np.sqrt(2)
    wires = [Wire(0, 0, rw)]
    ref = Wire(x2, y2, rw)
    pair = mtl.WireMtl(wires, ref)
    C = mtl.wire_capacitance(s, rw)
    assert pair.capacitance()[[0]] == approx(C, rel=0.02, abs=1e-14)


def test_two_wire_capacitance_er():
    er = 4
    rw = 7.5
    s = 50
    wires = [Wire(-0.5 * s, 0, rw)]
    ref = Wire(0.5 * s, 0, rw)
    pair = mtl.WireMtl(wires, ref, er)
    C = mtl.wire_capacitance(s, rw, er=er)
    assert pair.capacitance()[[0]] == approx(C, rel=0.02, abs=1e-14)


def test_two_wire_inductance():
    rw = 7.5
    s = 50
    wires = [Wire(-0.5 * s, 0, rw)]
    ref = Wire(0.5 * s, 0, rw)
    pair = mtl.WireMtl(wires, ref)
    L = mtl.wire_inductance(s, rw)
    assert pair.inductance()[[0]] == approx(L, rel=0.02)


def test_two_wire_inductance_diagonal():
    rw = 7.5
    s = 50
    x2 = y2 = s / np.sqrt(2)
    wires = [Wire(0, 0, rw)]
    ref = Wire(x2, y2, rw)
    pair = mtl.WireMtl(wires, ref)
    L = mtl.wire_inductance(s, rw)
    assert pair.inductance()[[0]] == approx(L, rel=0.02)


def test_two_wire_capacitance_bad_method():
    rw = 1e-3
    s = 10e-3
    wires = [Wire(-0.5 * s, 0, rw)]
    ref = Wire(0.5 * s, 0, rw)
    pair = mtl.WireMtl(wires, ref)
    with pytest.raises(Exception):
        pair.capacitance(method='junk')


def test_two_wire_capacitance_fdm():
    rw = 1e-3
    s = 8e-3
    wires = [Wire(-0.5 * s, 0, rw)]
    ref = Wire(0.5 * s, 0, rw)
    pair = mtl.WireMtl(wires, ref)
    expected = pair.capacitance(method='ana')
    C = pair.capacitance(method='fdm')
    assert C == approx(expected, rel=0.05, abs=0.1e-12)


def test_two_wire_capacitance_fdm_diagonal():
    rw = 1e-3
    x2 = y2 = 6e-3
    wires = [Wire(0, 0, rw)]
    ref = Wire(x2, y2, rw)
    pair = mtl.WireMtl(wires, ref)
    expected = pair.capacitance(method='ana')
    C = pair.capacitance(method='fdm')
    assert C == approx(expected, rel=0.05, abs=0.1e-12)


def test_two_wire_capacitance_fdm_near():
    rw = 1e-3
    s = 3e-3
    wires = [Wire(-0.5 * s, 0, rw)]
    ref = Wire(0.5 * s, 0, rw)
    pair = mtl.WireMtl(wires, ref)
    expected = mtl.wire_capacitance(s, rw)
    C = pair.capacitance(method='fdm', fdm_params={'dx': rw / 10})
    assert C == approx(expected, rel=0.05, abs=0.1e-12)


@pytest.mark.xfail(strict=True)  # TODO adjust params for better accuracy
def test_two_wire_capacitance_fdm_rw2():
    rw1 = 1e-3
    rw2 = 2e-3
    s = 20e-3
    wires = [Wire(-0.5 * s, 0, rw1)]
    ref = Wire(0.5 * s, 0, rw2)
    pair = mtl.WireMtl(wires, ref)
    expected = pair.capacitance(method='ana')
    print(expected)
    C = pair.capacitance(method='fdm')
    assert C == approx(expected, rel=0.05, abs=0.1e-12)


def test_three_wire_inductance():
    # Paul MTL P5.4
    rw = 7.5
    s = 50
    wires = [Wire(s, 0, rw), Wire(-s, 0, rw)]
    ref = Wire(0, 0, rw)
    bus = mtl.WireMtl(wires, ref)
    L = bus.inductance()
    expected = np.array([[0.759, 0.241], [0.241, 0.759]]) * 1e-6
    assert L == approx(expected, rel=0.001)


def test_three_wire_capacitance():
    # Paul MTL P5.4
    rw = 7.5
    s = 50
    wires = [Wire(s, 0, rw), Wire(-s, 0, rw)]
    ref = Wire(0, 0, rw)
    bus = mtl.WireMtl(wires, ref)
    C = bus.capacitance()
    expected = np.array([[16.3, -5.17], [-5.17, 16.3]]) * 1e-12
    assert C == approx(expected, rel=0.001, abs=1e-14)


def test_two_wire_inductance_plane():
    # Paul MTL P5.7
    rw = 0.04064
    s = 2
    h = 2
    wires = [Wire(-0.5 * s, h, rw), Wire(0.5 * s, h, rw)]
    ref = Plane()
    pair = mtl.WireMtl(wires, ref)
    L = pair.inductance()
    expected = np.array([[0.918, 0.161], [0.161, 0.918]]) * 1e-6
    assert L == approx(expected, rel=0.001)


def test_two_wire_capacitance_plane():
    # Paul MTL P5.7
    rw = 0.04064
    s = 2
    h = 2
    wires = [Wire(-0.5 * s, h, rw), Wire(0.5 * s, h, rw)]
    ref = Plane()
    pair = mtl.WireMtl(wires, ref)
    C = pair.capacitance()
    expected = np.array([[12.5, -2.19], [-2.19, 12.5]]) * 1e-12
    assert C == approx(expected, rel=0.001, abs=1e-14)


def test_two_wire_inductance_shield():
    # Paul MTL P5.10
    rw = 0.1905e-3
    s = 4 * rw
    rs = 4 * rw
    wires = [Wire(-0.5 * s, 0, rw), Wire(0.5 * s, 0, rw)]
    ref = Shield(rs)
    pair = mtl.WireMtl(wires, ref)
    L = pair.inductance()
    expected = np.array([[0.2197, 0.0446], [0.0446, 0.2197]]) * 1e-6
    assert L == approx(expected, rel=0.001)


def test_two_wire_capacitance_shield():
    # Paul MTL P5.11
    rw = 0.1905e-3
    s = 4 * rw
    rs = 4 * rw
    wires = [Wire(-0.5 * s, 0, rw), Wire(0.5 * s, 0, rw)]
    ref = Shield(rs)
    pair = mtl.WireMtl(wires, ref)
    C = pair.capacitance()
    expected = np.array([[52.8, -10.73], [-10.73, 52.8]]) * 1e-12
    assert C == approx(expected, rel=0.001, abs=1e-14)


def test_one_wire_capacitance_shield():
    rw = 0.5e-3
    rs = 4e-3
    er = 5.2
    wires = [Wire(0, 0, rw)]
    ref = Shield(rs)
    cable = mtl.WireMtl(wires, ref, er)
    C = cable.capacitance()
    expected = coax.capacitance(rw, rs, er)
    assert C == approx(expected, rel=0.001, abs=1e-14)
