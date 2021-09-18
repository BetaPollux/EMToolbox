#!/usr/bin/python3

from emtoolbox.tline.wire_mtl import WireMtl
from emtoolbox.tline.wire import Wire, Plane, Shield
import emtoolbox.external.fastcap2d as fc
import matplotlib.pyplot as plt


def wire_mtl_to_fastcap2d(mtl, filename):
    radii = set()
    elements = []
    wires = [*mtl.wires]
    if type(mtl.ref) == Wire:
        wires.append(mtl.ref)
    # Create wires
    for wire in wires:
        radii.add(wire.radius)
        elements.append(fc.Conductor(name_circ_file(wire.radius),
                        wire.ins_er, wire.x, wire.y))
        if wire.ins_thickness > 0:
            radii.add(wire.ins_radius)
            elements.append(fc.Dielectric(name_circ_file(wire.ins_radius),
                            1.0, wire.ins_er, wire.x, wire.y,
                            wire.x, wire.y, ref_is_inperm=True))
    # Create reference conductor
    if type(mtl.ref) == Plane:
        elements.append(fc.Conductor(name_rect_file(mtl.ref.width, mtl.ref.thickness),
                                     1.0, -0.5 * mtl.ref.width, -mtl.ref.thickness))
        box_seg = fc.generate_box_segments(mtl.ref.width, mtl.ref.thickness)
        fc.save_segments(name_rect_file(mtl.ref.width, mtl.ref.thickness), box_seg,
                         f'Box of {mtl.ref.width} by {mtl.ref.thickness} mm')
    elif type(mtl.ref) == Shield:
        radii.add(mtl.ref.radius)
        elements.append(fc.Conductor(name_circ_file(mtl.ref.radius),
                        1.0, 0, 0))
    # Generate files
    for radius in radii:
        N = 32
        circ_seg = fc.generate_arc_segments(radius, 0, 360, N=N)
        fn = name_circ_file(radius)
        fc.save_segments(fn, circ_seg,
                         f'Circle of {2000 * radius} mm diameter, {N} segments')
    fc.save_list_file(filename, elements,
                      f'Wire MTL, {len(mtl.wires) + 1} conductors')


def name_circ_file(radius: float):
    return f'circ_{2000 * radius}mm.txt'


def name_rect_file(width: float, height: float):
    return f'rect_{width}_{height}mm.txt'


def test_export():
    PITCH = 1.27e-3
    RW = 0.1905e-3
    INS_ER = 3.5
    INS_THK = 0.254e-3

    # CR Paul MTL Table 5.5
    # With / without dielectric
    # C11   37.432      22.494  pF/m
    # C12   -18.716     -11.247 pF/m
    # C22   24.982      16.581  pF/m
    capacitance = []
    options = ((0.0, 1.0, 'no_'),
               (INS_THK, INS_ER, ''))
    for (ins_thk, ins_er, prefix) in options:
        filename = f'ribbon_{prefix}dielectric.lst'
        ref = Wire(0, 0, RW, ins_thk, ins_er)
        # ref = Plane()
        # ref = Shield(3 * PITCH)
        wires = [Wire(PITCH, 0, RW, ins_thk, ins_er),
                 Wire(2 * PITCH, 0, RW, ins_thk, ins_er)]
        mtl = WireMtl(wires, ref)
        wire_mtl_to_fastcap2d(mtl, filename)
        try:
            capacitance.append(fc.run_fastercap(filename))
        except Exception as ex:
            print(ex)
        fc.plot_list_file(filename)
    if len(capacitance) == 2:
        print('*****')
        print('No Dielectric (pF/m):')
        print(capacitance[0] * 1e12)
        print('With Dielectric (pF/m):')
        print(capacitance[1] * 1e12)
    plt.show()


if __name__ == '__main__':
    test_export()
