import numpy as np
import pytest
from pytest import approx
import emtoolbox.tline.wire_mtl as mtl


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
    assert C == approx(Cer / er, abs=0.1e-12)


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


def test_one_conductor():
    with pytest.raises(Exception):
        mtl.WireMtl([(0, 0, 1)])


def test_bad_conductor_types():
    with pytest.raises(Exception):
        mtl.WireMtl([1, 1])


def test_bad_conductor_def():
    with pytest.raises(Exception):
        mtl.WireMtl([(0, 0, 1), (0, 1)])


def test_two_wire_capacitance():
    rw = 7.5
    s = 50
    wires = [(-0.5 * s, 0, rw), (0.5 * s, 0, rw)]
    pair = mtl.WireMtl(wires)
    C = mtl.wire_capacitance(s, rw)
    assert pair.capacitance()[0] == approx(C)


def test_two_wire_capacitance_diagonal():
    rw = 7.5
    s = 50
    x2 = y2 = s / np.sqrt(2)
    wires = [(0, 0, rw), (x2, y2, rw)]
    pair = mtl.WireMtl(wires)
    C = mtl.wire_capacitance(s, rw)
    assert pair.capacitance()[0] == approx(C)


def test_two_wire_capacitance_er():
    er = 4
    rw = 7.5
    s = 50
    wires = [(-0.5 * s, 0, rw), (0.5 * s, 0, rw)]
    pair = mtl.WireMtl(wires, er)
    C = mtl.wire_capacitance(s, rw, er=er)
    assert pair.capacitance()[0] == approx(C)


def test_two_wire_inductance():
    rw = 7.5
    s = 50
    wires = [(-0.5 * s, 0, rw), (0.5 * s, 0, rw)]
    pair = mtl.WireMtl(wires)
    L = mtl.wire_inductance(s, rw)
    assert pair.inductance()[0] == approx(L)


def test_two_wire_inductance_diagonal():
    rw = 7.5
    s = 50
    x2 = y2 = s / np.sqrt(2)
    wires = [(0, 0, rw), (x2, y2, rw)]
    pair = mtl.WireMtl(wires)
    L = mtl.wire_inductance(s, rw)
    assert pair.inductance()[0] == approx(L)


def test_two_wire_capacitance_bad_method():
    rw = 1e-3
    s = 10e-3
    wires = [(-0.5 * s, 0, rw), (0.5 * s, 0, rw)]
    pair = mtl.WireMtl(wires)
    with pytest.raises(Exception):
        pair.capacitance(method='junk')


def test_two_wire_capacitance_fdm():
    rw = 1e-3
    s = 8e-3
    wires = [(-0.5 * s, 0, rw), (0.5 * s, 0, rw)]
    pair = mtl.WireMtl(wires)
    expected = pair.capacitance(method='ana')
    C = pair.capacitance(method='fdm')
    assert C == approx(expected, rel=0.05, abs=0.1e-12)


def test_two_wire_capacitance_fdm_diagonal():
    rw = 1e-3
    x2 = y2 = 6e-3
    wires = [(0, 0, rw), (x2, y2, rw)]
    pair = mtl.WireMtl(wires)
    expected = pair.capacitance(method='ana')
    C = pair.capacitance(method='fdm')
    assert C == approx(expected, rel=0.05, abs=0.1e-12)


def test_two_wire_capacitance_fdm_near():
    rw = 1e-3
    s = 3e-3
    wires = [(-0.5 * s, 0, rw), (0.5 * s, 0, rw)]
    pair = mtl.WireMtl(wires)
    expected = pair.capacitance(method='ana')
    C = pair.capacitance(method='fdm', fdm_params={'dx': rw / 12})
    assert C == approx(expected, rel=0.05, abs=0.1e-12)


@pytest.mark.xfail(strict=True)  # TODO adjust params for better accuracy
def test_two_wire_capacitance_fdm_rw2():
    rw1 = 1e-3
    rw2 = 2e-3
    s = 20e-3
    wires = [(-0.5 * s, 0, rw1), (0.5 * s, 0, rw2)]
    pair = mtl.WireMtl(wires)
    expected = pair.capacitance(method='ana')
    print(expected)
    C = pair.capacitance(method='fdm')
    assert C == approx(expected, rel=0.05, abs=0.1e-12)
