#! /usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Grid1D:
    def __init__(self, n, width):
        self.x = np.linspace(0, width, n)
        self.ex = np.zeros(n)
        self.hy = np.zeros(n)
        self.sources = []
        self.data = []

    def solve(self, num_steps):
        self.t = np.arange(num_steps)

        for time in self.t:
            for xi in range(1, len(self.ex)):
                self.ex[xi] = self.ex[xi] + 0.5*(self.hy[xi - 1] - self.hy[xi])

            for source in self.sources:
                self.ex[source['position']] += source['source'].solve(time)

            for xi in range(len(self.hy) - 1):
                self.hy[xi] = self.hy[xi] + 0.5*(self.ex[xi] - self.ex[xi + 1])

            self.data.append(self.ex.copy())

    def add_source(self, source, x):
        self.sources.append({'source': source,
                             'position': x
                             })


class Gaussian:
    def __init__(self, spread, t0):
        self.spread = spread
        self.t0 = t0

    def solve(self, time):
        return np.exp(-0.5 * ((self.t0 - time)/self.spread) ** 2.0)


def animate(i, *fargs):
    line = fargs[0]
    data = fargs[1]
    line.set_ydata(data[i])
    return line,


def main():
    ndt = 500
    ndx = 200
    grid = Grid1D(ndx, 1.0)
    grid.add_source(Gaussian(12.0, 40.0), ndx//2)
    grid.solve(ndt)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(grid.ex)
    axes[0].set_ylabel('Ex')
    axes[1].plot(grid.hy)
    axes[1].set_ylabel('Hy')

    fig2, ax2 = plt.subplots()
    ax2.set_ylim(-2, 2)
    eline, = ax2.plot(grid.ex)
    ani = animation.FuncAnimation(
        fig2, animate, fargs=(eline, grid.data), interval=20,
        blit=True, frames=ndt)

    plt.show()


if __name__ == '__main__':
    main()
