#!/usr/bin/python3

'''Classes to handle FastCap 2D files'''

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import subprocess
import re


class Segment():
    def __init__(self, name: str, x1: float, y1: float, x2: float, y2: float):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __repr__(self):
        return f'S {self.name} {self.x1} {self.y1} {self.x2} {self.y2}'

    def offset(self, dx, dy):
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy

    @classmethod
    def make_segment(cls, statement: str):
        tokens = statement.split()
        if tokens[0] != 'S':
            raise ValueError('Invalid statement; not an S statement')
        if len(tokens) < 6:
            raise ValueError('Invalid statement; insufficient values')
        name = tokens[1]
        values = [float(token) for token in tokens[2:]]
        return cls(name, *values)

    @classmethod
    def read_file(cls, filename: str):
        return read_fastcap2d_file(filename).get('S', [])


class Conductor():
    def __init__(self, file: str, outperm: float, xoffset: float,
                 yoffset: float, merge: bool = False):
        self.file = file
        self.outperm = outperm
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.merge = merge

    def __repr__(self):
        return f'C {self.file} {format_permittivity(self.outperm)} {self.xoffset} {self.yoffset}{" +" if self.merge else ""}'

    def get_segments(self):
        segments = Segment.read_file(self.file)
        offset_segments(segments, self.xoffset, self.yoffset)
        return segments

    @classmethod
    def make_conductor(cls, statement: str):
        tokens = statement.split()
        if tokens[0] != 'C':
            raise ValueError('Invalid statement; not an C statement')
        if len(tokens) < 5:
            raise ValueError('Invalid statement; insufficient values')
        file = tokens[1]
        if tokens[-1] == '+':
            merge = True
            tokens.pop()
        else:
            merge = False
        outperm = parse_permittivity(tokens[2])
        values = [float(token) for token in tokens[3:]]
        return cls(file, outperm, *values, merge)

    @classmethod
    def read_file(cls, filename: str):
        return read_fastcap2d_file(filename).get('C', [])


class Dielectric():
    def __init__(self, file: str, outperm: float, inperm: float,
                 xoffset: float, yoffset: float, xref: float, yref: float,
                 ref_is_inperm: bool = True):
        self.file = file
        self.outperm = outperm
        self.inperm = inperm
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.xref = xref
        self.yref = yref
        self.ref_is_inperm = ref_is_inperm

    def __repr__(self):
        return f'D {self.file} {format_permittivity(self.outperm)} {format_permittivity(self.inperm)} {self.xoffset} {self.yoffset} {self.xref} {self.yref}{" -" if self.ref_is_inperm else ""}'

    def get_segments(self):
        segments = Segment.read_file(self.file)
        offset_segments(segments, self.xoffset, self.yoffset)
        return segments

    @classmethod
    def make_dielectric(cls, statement: str):
        tokens = statement.split()
        if tokens[0] != 'D':
            raise ValueError('Invalid statement; not a D statement')
        if len(tokens) < 8:
            raise ValueError('Invalid statement; insufficient values')
        file = tokens[1]
        outperm = parse_permittivity(tokens[2])
        inperm = parse_permittivity(tokens[3])
        values = [float(token) for token in tokens[4:8]]
        ref_is_inperm = (tokens[-1] == '-')
        return cls(file, outperm, inperm, *values, ref_is_inperm)

    @classmethod
    def read_file(cls, filename: str):
        return read_fastcap2d_file(filename).get('D', [])


def read_fastcap2d_file(filename: str):
    elements = {}
    with open(filename, 'r') as fn:
        # Discard first line
        for line in fn.readlines()[1:]:
            if line.startswith('*') or line.isspace():
                continue
            elif line.startswith('S'):
                elements.setdefault('S', [])
                elements['S'].append(Segment.make_segment(line))
            elif line.startswith('C'):
                elements.setdefault('C', [])
                elements['C'].append(Conductor.make_conductor(line))
            elif line.startswith('D'):
                elements.setdefault('D', [])
                elements['D'].append(Dielectric.make_dielectric(line))
            else:
                print('Ignored line:', line)
    return elements


def parse_permittivity(text: str):
    '''FastCap format is 4.0-j3.016e6, Python needs 4.0-3.016e6j'''
    if 'j' in text:
        text = text.replace('j', '') + 'j'
        return complex(text)
    else:
        return float(text)


def format_permittivity(perm):
    if type(perm) is float:
        return str(perm)
    elif type(perm) is complex:
        return str(perm.real) + \
               ('+' if perm.imag > 0 else '-') + \
               'j' + str(abs(perm.imag))


def offset_segments(segments, dx, dy):
    for segment in segments:
        segment.offset(dx, dy)


def generate_arc_segments(radius: float, start: float, end: float,
                          N: int = 16):
    assert abs(start) <= 360 and abs(end) <= 360
    if start > end:
        end += 360
    angles = np.linspace(np.deg2rad(start), np.deg2rad(end), N)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    segments = []
    for i in range(1, N):
        segments.append(Segment('arc', x[i-1], y[i-1], x[i], y[i]))
    return segments


def generate_box_segments(width: float, height: float):
    return [
        Segment('box', width, 0.0, width, height),
        Segment('box', width, height, 0.0, height),
        Segment('box', 0.0, height, 0.0, 0.0),
        Segment('box', 0.0, 0.0, width, 0.0)
    ]


def save_segments(filename: str, segments: list, description: str = ''):
    with open(filename, 'w') as fn:
        fn.write('* 2D FastCap geometry file generated by fastcap2d.py\n')
        if description:
            fn.write('* ' + description + '\n')
        for segment in segments:
            fn.write(str(segment) + '\n')


def save_list_file(filename: str, elements, description: str = ''):
    with open(filename, 'w') as fn:
        fn.write('* 2D FastCap list file generated by fastcap2d.py\n')
        if description:
            fn.write('* ' + description + '\n')
        for element in elements:
            fn.write(str(element) + '\n')


def plot_segments(ax, segments, color='b', ls='solid'):
    segs = np.array([[[s.x1, s.y1], [s.x2, s.y2]] for s in segments])
    lines = LineCollection(segs, colors=color, ls=ls)
    ax.add_collection(lines)


def plot_label(ax, segments, text, color='b'):
    x = min([min(s.x1, s.x2) for s in segments])
    y = max([max(s.y1, s.y2) for s in segments])
    ax.text(x, y, text, color=color, ha='left', va='bottom')


def plot_conductors(ax, conductors, labels=True):
    cid = 1
    cid_sub = 'a'
    for conductor in conductors:
        segments = conductor.get_segments()
        text = f'C{cid}'
        if conductor.merge:
            # Split conductor, do not increment id, append a, b, c, etc.
            text += cid_sub
            cid_sub = chr(ord(cid_sub) + 1)
        else:
            if cid_sub > 'a':  # Last element of split conductor
                text += cid_sub
            cid += 1
            cid_sub = 'a'
        if conductor.outperm == 1.0:
            ls = 'solid'
        else:
            ls = 'dashed'
            text += f'; {conductor.outperm}'
        plot_segments(ax, segments, color='b', ls=ls)
        if labels:
            plot_label(ax, segments, text, color='b')


def plot_dielectrics(ax, dielectrics, labels=True):
    for dielectric in dielectrics:
        segments = dielectric.get_segments()
        plot_segments(ax, segments, color='r', ls='dotted')
        if labels:
            if dielectric.ref_is_inperm:
                at_ref = f'{dielectric.inperm}'
                at_mid = f'{dielectric.outperm}'
            else:
                at_ref = f'{dielectric.outperm}'
                at_mid = f'{dielectric.inperm}'
            ax.text(dielectric.xref, dielectric.yref, at_ref, color='r',
                    ha='center', va='center')
            plot_label(ax, segments, at_mid, color='r')


def plot_list_file(filename):
    elements = read_fastcap2d_file(filename)
    fig, ax = plt.subplots()
    fig.suptitle(filename)
    plot_conductors(ax, elements.get('C', []))
    plot_dielectrics(ax, elements.get('D', []))
    ax.autoscale()
    return ax


def run_fastercap(list_file: str,
                  exe: str = r'C:\Program Files (x86)\FastFieldSolvers\FasterCap\FasterCap.exe'):
    output = subprocess.run([exe, f'-b {list_file} -a0.01'],
                            capture_output=True, text=True)
    if output.returncode != 0:
        raise RuntimeError(
            f'FasterCap exited with error code {output.returncode}')
    last_iter = output.stdout.rfind('Iteration number #')
    result = output.stdout[last_iter:]
    dim = re.search(r"^Dimension (\d+) x (\d+)",
                    result, flags=re.MULTILINE)
    matrix_iter = re.finditer(r"^g(\d+)_\S+\s+(.+)",
                              result, flags=re.MULTILINE)
    if dim is None or matrix_iter is None:
        raise ValueError('FasterCap run was unsuccessful')
    capacitance = np.zeros((int(dim.group(1)), int(dim.group(2))))
    for line in matrix_iter:
        row = int(line.group(1)) - 1
        # TODO this does not handle complex valued results
        capacitance[row, :] = [float(col) for col in line.group(2).split()]
    return capacitance


if __name__ == '__main__':
    print('FastCap2d')
    seg = Segment.make_segment('S inner_shell 0.098768907 0.015642989  0.095105938 0.030900818 ')
    print(seg)
    segs = Segment.read_file('circle_0.1.txt')
    c1 = Conductor.make_conductor('C microstrip_top_0.03.txt 1.0  -0.025 0.01 + ')
    c2 = Conductor.make_conductor('C microstrip_bottom_0.03.txt 2.0  -0.025 0.01 ')
    c3 = Conductor.make_conductor('C circle_0.005.txt 4.0-j3.016e6  0.0 0.02')
    print(c1)
    print(c2)
    print(c3)

    arc1 = generate_arc_segments(1.0, -45, 45, N=16)
    arc2 = generate_arc_segments(1.5, 180, 90, N=32)
    arc3 = generate_arc_segments(0.5, 0, 360, N=32)
    save_segments('box.txt', generate_box_segments(1.0, 0.5),
                  'Rectangular box, 1.0 m x 0.5 m')
    box = Segment.read_file('box.txt')

    print(run_fastercap('three_line_bus_2d.lst'))

    fig, ax = plt.subplots()
    plot_segments(ax, arc1, color='b')
    plot_segments(ax, arc2, color='r')
    plot_segments(ax, arc3, color='g')
    plot_segments(ax, box, color='k')
    ax.set_aspect(1.0)
    ax.autoscale()
    plt.show()

    plot_list_file('coax_cable_coated_2d_fine.lst')
    plot_list_file('coupled_microstrips_5x_2d.lst')
    plot_list_file('coupled_striplines_with_gaps_2d.lst')
    plot_list_file('circular_wire_over_gnd_plane_2d_lossy.lst')
    plot_list_file('three_line_bus_2d.lst')
    plt.show()
