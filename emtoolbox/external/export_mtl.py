#!/usr/bin/python3

from emtoolbox.tline.wire_mtl import WireMtl
from emtoolbox.tline.wire import Wire
import emtoolbox.external.fastcap2d as fc
import matplotlib.pyplot as plt


def wire_mtl_to_fastcap2d(mtl, filename):
    assert type(mtl.ref) == Wire  # TODO support Plane and Shield
    wires = [*mtl.wires, mtl.ref]
    radii = set()
    elements = []
    for wire in wires:
        radii.add(wire.radius)
        elements.append(fc.Conductor(name_wire_file(wire.radius),
                        wire.ins_er, wire.x, wire.y))
        if wire.ins_thickness > 0:
            radii.add(wire.ins_radius)
            elements.append(fc.Dielectric(name_wire_file(wire.ins_radius),
                            1.0, wire.ins_er, wire.x, wire.y,
                            wire.x, wire.y, ref_is_inperm=True))
    for radius in radii:
        N = 32
        seg = fc.generate_arc_segments(radius, 0, 360, N=N)
        fn = name_wire_file(radius)
        fc.save_segments(fn, seg,
                         f'Circle of {2000 * radius} mm diameter, {N} segments')
    fc.save_list_file(filename, elements,
                      f'Wire MTL, {len(mtl.wires) + 1} conductors')


def name_wire_file(radius: float):
    return f'circ_{2000 * radius}mm.txt'


def test_export():
    PITCH = 1.27e-3
    RW = 0.1905e-3
    INS_ER = 3.5
    INS_THK = 0.254e-3

    # With / without dielectric
    # C11   37.432      22.494  pF/m
    # C12   -18.716     -11.247 pF/m
    # C22   24.982      16.581  pF/m

    options = ((0.0, 1.0, 'no_'),
               (INS_THK, INS_ER, ''))
    for (ins_thk, ins_er, prefix) in options:
        filename = f'ribbon_{prefix}dielectric.lst'
        ref = Wire(0, 0, RW, ins_thk, ins_er)
        wires = [Wire(PITCH, 0, RW, ins_thk, ins_er),
                 Wire(2 * PITCH, 0, RW, ins_thk, ins_er)]
        mtl = WireMtl(wires, ref)
        wire_mtl_to_fastcap2d(mtl, filename)
        fc.plot_list_file(filename)
    plt.show()


if __name__ == '__main__':
    test_export()
