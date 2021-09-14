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

    def hit(self, x, y):
        for child in reversed(self.children):
            if child.hit(x, y):
                return child.positive
        return False

    def bounds(self):
        if len(self.children) == 0:
            raise Exception('Geometry has no children')
        left = self.children[0].left
        right = self.children[0].right
        bottom = self.children[0].bottom
        top = self.children[0].top
        for child in self.children:
            left = min(left, child.left)
            right = max(right, child.right)
            bottom = min(bottom, child.bottom)
            top = max(top, child.top)
        return (left, right, bottom, top)

    def grid(self, Nx, Ny, edges=True):
        xa, xb, ya, yb = self.bounds()
        x = np.linspace(xa, xb, Nx)
        y = np.linspace(ya, yb, Ny)
        if edges:
            return np.meshgrid(x, y, indexing='ij')
        else:
            return np.meshgrid(0.5*(x[1:] + x[:-1]), 0.5*(y[1:] + y[:-1]), indexing='ij')

    def mask(self, grid_x, grid_y):
        assert grid_x.shape == grid_y.shape
        x = 0.5 * (grid_x[1:, 0] + grid_x[:-1, 0])
        y = 0.5 * (grid_y[0, 1:] + grid_y[0, :-1])
        result = np.zeros((len(x), len(y)))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                result[i, j] = self.hit(xi, yi)
        return result

    def child_at(self, x, y):
        for child in reversed(self.children):
            if child.hit(x, y):
                return child
        return None

    def select(self, param, grid_x, grid_y):
        assert grid_x.shape == grid_y.shape
        x = 0.5 * (grid_x[1:, 0] + grid_x[:-1, 0])
        y = 0.5 * (grid_y[0, 1:] + grid_y[0, :-1])
        result = np.zeros((len(x), len(y)))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                result[i, j] = self.child_at(xi, yi).params.get(param, 0)
        return result


class Shape():
    def __init__(self):
        self.positive = True
        self.name = 'Shape'
        self.params = {}

    def hit(self, x, y) -> bool:
        return False


class Rect(Shape):
    def __init__(self, left: float, width: float, top: float, height: float):
        super().__init__()
        self.name = 'Rect'
        self.left = left
        self.width = width
        self.top = top
        self.height = height

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top - self.height

    def hit(self, x, y) -> bool:
        in_x = x >= self.left and x <= self.right
        in_y = y >= self.bottom and y <= self.top
        return in_x and in_y


class Circ(Shape):
    def __init__(self, mid_x: float, mid_y: float, radius: float):
        super().__init__()
        self.name = 'Circ'
        self.mid_x = mid_x
        self.mid_y = mid_y
        self.radius = radius

    @property
    def left(self):
        return self.mid_x - self.radius

    @property
    def top(self):
        return self.mid_y + self.radius

    @property
    def right(self):
        return self.mid_x + self.radius

    @property
    def bottom(self):
        return self.mid_y - self.radius

    def hit(self, x, y) -> bool:
        dx2 = (self.mid_x - x) ** 2
        dy2 = (self.mid_y - y) ** 2
        return dx2 + dy2 <= self.radius ** 2


if __name__ == '__main__':
    rectp = Rect(0.0, 4.0, 6.0, 6.0)
    rectn = Rect(1.0, 2.0, 3.0, 2.0)
    rectp2 = Rect(1.5, 1.0, 4.0, 2.0)
    circ = Circ(3.0, 2.0, 0.5)
    rectn.positive = False
    rectp.params['er'] = 2.0
    rectp2.params['er'] = 4.0
    circ.params['er'] = 3.0
    geom = Geometry()
    geom.add_child(rectp)
    geom.add_child(rectn)
    geom.add_child(rectp2)
    geom.add_child(circ)

    Nx, Ny = 81, 121
    X, Y = geom.grid(Nx, Ny)
    print(X)
    print(Y)
    b = geom.mask(X, Y)
    e = geom.select('er', X, Y)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.pcolor(X, Y, b, shading='auto')
    ax1.set_ylabel('Mask')
    ax2.pcolor(X, Y, e, shading='auto')
    ax2.set_ylabel('er')
    plt.show()
