#/bin/usr/python3

'''Classes to handle FastCap 2D files'''

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


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
            raise ValueError ('Invalid statement; insufficient values')
        name = tokens[1]
        values = [float(token) for token in tokens[2:]]
        return cls(name, *values)
    
    @classmethod
    def read_file(cls, filename: str):
        segments = []
        with open(filename, 'r') as fn:
            # Discard first line
            for line in fn.readlines()[1:]:
                if line.startswith('*') or line.isspace():
                    continue
                else:
                    segments.append(cls.make_segment(line))
        return segments


class Conductor():
    def __init__(self, file: str, outperm: float, xoffset: float, yoffset: float, merge: bool):
        self.file = file
        self.outperm = outperm
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.merge = merge
        self.segments = []
    
    def __repr__(self):
        return f'C {self.file} {self.outperm} {self.xoffset} {self.yoffset}{" +" if self.merge else ""}'

    def get_segments(self):
        segments = Segment.read_file(self.file)
        for segment in segments:
            segment.offset(self.xoffset, self.yoffset)
        return segments

    @classmethod
    def make_conductor(cls, statement: str):
        tokens = statement.split()
        if tokens[0] != 'C':
            raise ValueError('Invalid statement; not an C statement')
        if len(tokens) < 5:
            raise ValueError ('Invalid statement; insufficient values')
        file = tokens[1]
        if tokens[-1] == '+':
            merge = True
            tokens.pop()
        else:
            merge = False
        values = [float(token) for token in tokens[2:]]
        return cls(file, *values, merge)
    
    @classmethod
    def read_file(cls, filename: str):
        conductors = []
        with open(filename, 'r') as fn:
            # Discard first line
            for line in fn.readlines()[1:]:
                if line.startswith('*') or line.isspace():
                    continue
                elif line.startswith('C'):
                    conductors.append(cls.make_conductor(line))
                else:
                    print('Ignored line:', line)
        return conductors


def plot_segments(ax, segments, color='b'):
    segs = np.array([[[s.x1, s.y1], [s.x2, s.y2]] for s in segments])
    lines = LineCollection(segs, colors=color)
    ax.add_collection(lines)


def plot_conductors(ax, conductors, labels=False):
    for i, conductor in enumerate(conductors):
        segments = conductor.get_segments()
        plot_segments(ax, segments)
        if labels:
            avgx = sum([s.x1 + s.x2 for s in segments]) / (2 * len(segments))
            avgy = sum([s.y1 + s.y2 for s in segments]) / (2 * len(segments))
            ax.text(avgx, avgy, f'{i + 1}', ha='center', va='center')


if __name__ == '__main__':
    print('FastCap2d')
    seg = Segment.make_segment('S inner_shell 0.098768907 0.015642989  0.095105938 0.030900818 ')
    print(seg)
    segs = Segment.read_file('circle_0.1.txt')
    c1 = Conductor.make_conductor('C microstrip_top_0.03.txt 1.0  -0.025 0.01 + ')
    c2 = Conductor.make_conductor('C microstrip_bottom_0.03.txt 2.0  -0.025 0.01 ')
    print(c1)
    print(c2)
    conductors = Conductor.read_file('coupled_microstrips_5x_2d.lst')
    fig, ax = plt.subplots()
    plot_conductors(ax, conductors)
    ax.autoscale()
    plt.show()
