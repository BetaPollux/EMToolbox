#!/usr/bin/python3

'''Geometry based on a sequence of primitives
Primitives can be positive or negative shapes
Intended for generation of grid cells or meshes'''

import numpy as np
import matplotlib.pyplot as plt


class Geometry():
    def __init__(self):
        self.children = []

    def add_child(self, new_child):
        self.children.append(new_child)

    def hit(self, x):
        for child in reversed(self.children):
            if child.hit(x):
                return child.positive
        return False

    def bounds(self):
        if len(self.children) == 0:
            raise Exception('Geometry has no children')
        left = self.children[0].left
        right = self.children[0].right
        for child in self.children:
            left = min(left, child.left)
            right = max(right, child.right)
        return (left, right)

    def grid(self, N, edges=True):
        xa, xb = self.bounds()
        x = np.linspace(xa, xb, N)
        if edges:
            return x
        else:
            return 0.5*(x[1:] + x[:-1])

    def mask(self, grid):
        x = 0.5 * (grid[1:] + grid[:-1])
        mask = np.zeros_like(x)
        for i, xi in enumerate(x):
            mask[i] = self.hit(xi)
        return mask

    def child_at(self, x):
        for child in reversed(self.children):
            if child.hit(x):
                return child
        return None

    def select(self, param, grid):
        x = 0.5 * (grid[1:] + grid[:-1])
        result = np.zeros_like(x)
        for i, xi in enumerate(x):
            result[i] = self.child_at(xi).params.get(param, 0)
        return result


class Shape():
    def __init__(self):
        self.positive = True
        self.name = 'Shape'
        self.params = {}

    def hit(self, x) -> bool:
        return False


class Rect(Shape):
    def __init__(self, left: float, width: float):
        super().__init__()
        self.name = 'Rect'
        self.left = left
        self.width = width

    @property
    def right(self):
        return self.left + self.width

    def hit(self, x) -> bool:
        return x >= self.left and x <= self.right


if __name__ == '__main__':
    rectp = Rect(0.0, 4.0)
    rectn = Rect(1.0, 2.0)
    rectp2 = Rect(1.5, 1.0)
    rectn.positive = False
    rectp.params['er'] = 2.0
    rectp2.params['er'] = 4.0
    geom = Geometry()
    geom.add_child(rectp)
    geom.add_child(rectn)
    geom.add_child(rectp2)

    N = 9
    x = geom.grid(N)
    y = np.array([0.0, 1.0])
    X, Y = np.meshgrid(x, y, indexing='ij')
    b = geom.mask(x)
    e = geom.select('er', x)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.pcolor(X, Y, b[:, np.newaxis], shading='auto')
    ax1.set_ylabel('Mask')
    ax2.pcolor(X, Y, e[:, np.newaxis], shading='auto')
    ax2.set_ylabel('er')
    plt.show()
