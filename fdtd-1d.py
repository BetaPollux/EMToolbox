#! /usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

eps0 = 8.8541878176e-12


class Grid1D:
    def __init__(self, ndz, width):
        self.ndz = ndz
        self.z, self.dz = np.linspace(0, width, ndz, retstep=True)
        self.dt = self.dz / (2 * 3e8)
        self.ex = np.zeros(ndz)
        self.hy = np.zeros(ndz)
        self.ca = np.ones(ndz)
        self.cb = 0.5 * np.ones(ndz)
        self.sources = []
        self.data = []

    def set_material(self, start, stop, er=1.0, cond=0.0):
        indices = (self.z >= start) & (self.z <= stop)
        eaf = self.dt * cond / (2 * eps0 * er)
        self.ca[indices] = (1 - eaf) / (1 + eaf)
        self.cb[indices] = 0.5 / (er * (1 + eaf))

    def solve(self, total_time, n_frames=125):
        self.t = np.arange(total_time, step=self.dt)
        frame_ids = np.linspace(0, len(self.t) - 1,
                                int(n_frames), dtype='int64')

        abc_left = [0, 0]
        abc_right = [0, 0]

        for time_id, time in enumerate(self.t):
            for i in range(1, self.ndz):
                self.ex[i] = self.ca[i] * self.ex[i] + self.cb[i] * (
                             self.hy[i - 1] - self.hy[i])

            for source in self.sources:
                self.ex[source['position']] += source['source'].solve(time)

            self.ex[0] = abc_left.pop()
            abc_left.insert(0, self.ex[1])
            self.ex[-1] = abc_right.pop()
            abc_right.insert(0, self.ex[-2])

            for i in range(self.ndz - 1):
                self.hy[i] = self.hy[i] + 0.5 * (
                             self.ex[i] - self.ex[i + 1])

            if time_id in frame_ids:
                self.data.append(self.ex.copy())

    def add_source(self, source, z):
        self.sources.append({'source': source,
                             'position': z
                             })


class Gaussian:
    def __init__(self, amplitude, t0, spread):
        self.amplitude = amplitude
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


class SinusoidalGauss:
    """
    Sinusoidally-modulated Gaussian Pulse
    f1 = lower frequency limit
    f2 = upper frequency limit
    bbw = minimum pulse level that is unaffected by computational noise
    bt = maximum allowable pulse level at t = 0
    bf = maximum allowable DC component in frequency domain
    """
    def __init__(self, f1, f2, bbw=0.0001, bt=0.001, bf=0.001):
        self.fc = 0.5 * (f1 + f2)
        self.alpha = np.pi * (f2 - f1) / np.sqrt(-np.log(bbw))
        self.t0 = np.sqrt(-np.log(bt)) / self.alpha

        if self.fc < self.alpha * np.pi / np.sqrt(-np.log(bf)):
            raise Exception('DC component exceeds allowable limit bf')

    def solve(self, time):
        return np.exp(-(self.alpha ** 2) * (time - self.t0) ** 2) \
               * np.cos(2 * np.pi * self.fc * (time - self.t0))


def animate(i, *fargs):
    line = fargs[0]
    data = fargs[1]
    line.set_ydata(data[i])
    return line,


def main():
    total_time = 20e-9
    n_frames = 250
    ndz = 200
    grid = Grid1D(ndz, 2.0)
    grid.set_material(1.0, 1.5, er=4.0, cond=0.04)
    # source = Gaussian(1.0, 40 * grid.dt, 12 * grid.dt)
    # source = Sinusoid(1.0, 700e6)
    source = SinusoidalGauss(1e9, 2e9)
    grid.add_source(source, 5)
    grid.solve(total_time, n_frames)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(grid.z, grid.ex)
    axes[0].set_ylabel('Ex')
    axes[1].plot(grid.z, grid.hy)
    axes[1].set_ylabel('Hy')
    axes[1].set_xlabel('z (m)')

    fig2, ax2 = plt.subplots()
    ax2.set_ylim(-2, 2)
    eline, = ax2.plot(grid.z, grid.ex)
    ax2.set_ylabel('Ex')
    ax2.set_xlabel('z (m)')
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
