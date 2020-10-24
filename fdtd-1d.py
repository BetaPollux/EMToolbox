#! /usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Grid1D:
    def __init__(self, ndx, width):
        self.ndx = ndx
        self.x, self.dx = np.linspace(0, width, ndx, retstep=True)
        self.dt = self.dx / (2 * 3e8)
        self.ex = np.zeros(ndx)
        self.hy = np.zeros(ndx)
        self.cb = 0.5 * np.ones(ndx)
        er = 4.0
        self.cb[self.x > width/2] /= er
        self.sources = []
        self.data = []

    def solve(self, total_time, n_frames=125):
        self.t = np.arange(total_time, step=self.dt)
        frame_ids = np.linspace(0, len(self.t) - 1,
                                int(n_frames), dtype='int64')

        abc_left = [0, 0]
        abc_right = [0, 0]

        for time_id, time in enumerate(self.t):
            for i in range(1, self.ndx):
                self.ex[i] = self.ex[i] + self.cb[i] * (
                              self.hy[i - 1] - self.hy[i])

            for source in self.sources:
                self.ex[source['position']] += source['source'].solve(time)

            self.ex[0] = abc_left.pop()
            abc_left.insert(0, self.ex[1])
            self.ex[-1] = abc_right.pop()
            abc_right.insert(0, self.ex[-2])

            for i in range(self.ndx - 1):
                self.hy[i] = self.hy[i] + 0.5 * (
                              self.ex[i] - self.ex[i + 1])

            if time_id in frame_ids:
                self.data.append(self.ex.copy())

    def add_source(self, source, x):
        self.sources.append({'source': source,
                             'position': x
                             })


class Gaussian:
    def __init__(self, amplitude, t0, spread):
        self.spread = spread
        self.t0 = t0

    def solve(self, time):
        return self.amplitude * np.exp(
               -0.5 * ((self.t0 - time)/self.spread) ** 2.0)


class Sinusoid:
    def __init__(self, amplitude, freq):
        self.freq = freq
        self.amplitude = amplitude

    def solve(self, time):
        return self.amplitude * np.sin(2 * np.pi * self.freq * time)


def animate(i, *fargs):
    line = fargs[0]
    data = fargs[1]
    line.set_ydata(data[i])
    return line,


def main():
    total_time = 8e-9
    n_frames = 125
    ndx = 200
    grid = Grid1D(ndx, 2.0)
    # source = Gaussian(1.0, 40 * grid.dt, 12 * grid.dt)
    source = Sinusoid(1.0, 700e6)
    grid.add_source(source, 5)
    grid.solve(total_time, n_frames)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(grid.x, grid.ex)
    axes[0].set_ylabel('Ex')
    axes[1].plot(grid.x, grid.hy)
    axes[1].set_ylabel('Hy')
    axes[1].set_xlabel('Position (m)')

    fig2, ax2 = plt.subplots()
    ax2.set_ylim(-2, 2)
    eline, = ax2.plot(grid.ex)
    ax2.set_ylabel('Ex')
    ax2.set_xlabel('Position (m)')
    ani = animation.FuncAnimation(
        fig2, animate, fargs=(eline, grid.data), interval=40,
        blit=True, frames=len(grid.data))

    fig3, ax3 = plt.subplots()
    ax3.plot(grid.t, source.solve(grid.t))
    ax3.set_title('Source')
    ax3.set_ylabel('Ex')
    ax3.set_xlabel('Time (s)')

    plt.show()


if __name__ == '__main__':
    main()
