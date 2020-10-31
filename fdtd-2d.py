#! /usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as collections
from matplotlib import cm
from scipy import fftpack

eps0 = 8.8541878176e-12


class Grid2D:
    """FDTD 2D Grid, TM Formulation"""
    def __init__(self, cellsize_x, width, length):
        self.cellsize_x = cellsize_x
        self.cellsize_y = cellsize_x    # Square cells
        self.x = np.arange(0.0, width, self.cellsize_x)
        self.y = np.arange(0.0, length, self.cellsize_y)
        self.ndx = len(self.x)
        self.ndy = len(self.y)
        self.dt = min(self.cellsize_x, self.cellsize_y) / (2 * 3e8)
        self.dz = np.zeros((self.ndx, self.ndy))
        self.ez = np.zeros((self.ndx, self.ndy))
        self.iz = np.zeros((self.ndx, self.ndy))
        self.hx = np.zeros((self.ndx, self.ndy))
        self.hy = np.zeros((self.ndx, self.ndy))
        self.ihx = np.zeros((self.ndx, self.ndy))
        self.ihy = np.zeros((self.ndx, self.ndy))
        self.ga = np.ones((self.ndx, self.ndy))
        self.gb = np.zeros((self.ndx, self.ndy))
        self.gi2 = np.ones(self.ndx)
        self.gi3 = np.ones(self.ndx)
        self.fi1 = np.zeros(self.ndx)
        self.fi2 = np.ones(self.ndx)
        self.fi3 = np.ones(self.ndx)
        self.gj2 = np.ones(self.ndy)
        self.gj3 = np.ones(self.ndy)
        self.fj1 = np.zeros(self.ndy)
        self.fj2 = np.ones(self.ndy)
        self.fj3 = np.ones(self.ndy)
        self.sources = []
        self.probes = []
        self.data = []

    def __repr__(self):
        s = 'ndx {0:} ndy {1:}\n'.format(self.ndx, self.ndy) \
            + 'x   {0:.3e} y {1:.3e}   m\n'.format(max(self.x), max(self.y)) \
            + 'dx  {0:.3e} dy {1:.3e}  m\n'.format(
                self.cellsize_x, self.cellsize_y) \
            + 'dt  {0:.3e}   s'.format(self.dt)
        return s

    def init_pml(self, npml):
        for i in range(npml+1):
            xnum = npml - i
            xd = npml
            xxn = xnum / xd
            xn = 0.25 * (xxn ** 3)
            self.gi2[i] = self.gi2[-1-i] = 1.0 / (1.0 + xn)
            self.gi3[i] = self.gi3[-1-i] = (1.0 - xn) / (1.0 + xn)
            self.gj2[i] = self.gj2[-1-i] = 1.0 / (1.0 + xn)
            self.gj3[i] = self.gj3[-1-i] = (1.0 - xn) / (1.0 + xn)

            xxn = (xnum - 0.5) / xd
            xn = 0.25 * (xxn ** 3)
            self.fi1[i] = self.fi1[-2-i] = xn
            self.fi2[i] = self.fi2[-2-i] = 1.0 / (1.0 + xn)
            self.fi3[i] = self.fi3[-2-i] = (1.0 - xn) / (1.0 + xn)

            self.fj1[i] = self.fj1[-2-i] = xn
            self.fj2[i] = self.fj2[-2-i] = 1.0 / (1.0 + xn)
            self.fj3[i] = self.fj3[-2-i] = (1.0 - xn) / (1.0 + xn)

    def set_material(self, p1, p2, er=1.0, cond=0.0):
        # indices = (self.x >= start[0]) & (self.x <= stop[0]) \
        #          & (self.y >= start[1]) & (self.y <= stop[1])
        self.ga[p1[0]:p2[0], p1[1]:p2[1]] = 1.0 / (er +
                                                   (cond * self.dt / eps0))
        self.gb[p1[0]:p2[0], p1[1]:p2[1]] = cond * self.dt / eps0

    def update_dz(self):
        for i in range(1, self.ndx):
            for j in range(1, self.ndy):
                self.dz[i, j] = self.gi3[i] * self.gj3[j] * self.dz[i, j] + (
                                self.gi2[i] * self.gj2[j] * 0.5 * (
                                    self.hy[i, j] -
                                    self.hy[i-1, j] -
                                    self.hx[i, j] +
                                    self.hx[i, j-1]))

    def update_ez(self):
        for i in range(1, self.ndx):
            for j in range(1, self.ndy):
                self.ez[i, j] = self.ga[i, j] * (self.dz[i, j] - self.iz[i, j])
                self.iz[i, j] = self.iz[i, j] + self.gb[i, j] * self.ez[i, j]

    def update_hx(self):
        for i in range(self.ndx - 1):
            for j in range(self.ndy - 1):
                curl_e = self.ez[i, j] - self.ez[i, j+1]
                self.ihx[i, j] = self.ihx[i, j] + self.fi1[i] * curl_e
                self.hx[i, j] = self.fj3[j] * self.hx[i, j] + (
                                self.fj2[j] * 0.5 * (curl_e + self.ihx[i, j]))

    def update_hy(self):
        for i in range(self.ndx - 1):
            for j in range(self.ndy - 1):
                curl_e = self.ez[i+1, j] - self.ez[i, j]
                self.ihy[i, j] = self.ihy[i, j] + self.fj1[i] * curl_e
                self.hy[i, j] = self.fi3[i] * self.hy[i, j] + (
                                self.fi2[i] * 0.5 * (curl_e + self.ihy[i, j]))

    def solve(self, total_time, n_frames=125):
        self.t = np.arange(total_time, step=self.dt)
        frame_ids = np.linspace(0, len(self.t) - 1,
                                int(n_frames), dtype='int64')
        for probe in self.probes:
            probe.data = np.zeros(len(self.t))

        print('Solving {0} time steps'.format(len(self.t)))
        print_ids = np.linspace(0, len(self.t) - 1, 11, dtype='int64')

        for time_id, time in enumerate(self.t):
            if time_id in print_ids:
                print('Step {0} {1:.0f}%'.format(
                    time_id,
                    100.0 * time_id / len(self.t)))

            self.update_dz()
            self.update_ez()

            for source in self.sources:
                self.ez[source.position[0],
                        source.position[1]] += source.solve(time)

            self.update_hx()
            self.update_hy()

            for probe in self.probes:
                probe.data[time_id] = self.ez[probe.position[0],
                                              probe.position[1]]

            if time_id in frame_ids:
                self.data.append(self.ez.copy())

        print('Solve complete')

    def add_source(self, source):
        self.sources.append(source)

    def add_probe(self, probe):
        self.probes.append(probe)


class Probe:
    def __init__(self, position, label='Probe'):
        self.position = position
        self.data = np.empty(1)
        self.label = label


class Source:
    def __init__(self, position, label='Source'):
        self.position = position
        self.label = label


class Gaussian(Source):
    def __init__(self, position, label, amplitude, t0, spread):
        super().__init__(position, label)
        self.amplitude = amplitude
        self.spread = spread
        self.t0 = t0

    def solve(self, time):
        return self.amplitude * np.exp(
               -0.5 * ((self.t0 - time)/self.spread) ** 2.0)


class Sinusoid(Source):
    def __init__(self, position, label, amplitude, freq):
        super().__init__(position, label)
        self.freq = freq
        self.amplitude = amplitude

    def solve(self, time):
        return self.amplitude * np.sin(2 * np.pi * self.freq * time)


class SinusoidalGauss(Source):
    """
    Sinusoidally-modulated Gaussian Pulse
    f1 = lower frequency limit
    f2 = upper frequency limit
    bbw = minimum pulse level that is unaffected by computational noise
    bt = maximum allowable pulse level at t = 0
    bf = maximum allowable DC component in frequency domain
    """
    def __init__(self, position, label, f1, f2,
                 bbw=0.0001, bt=0.001, bf=0.001):
        super().__init__(position, label)
        self.fc = 0.5 * (f1 + f2)
        self.alpha = np.pi * (f2 - f1) / np.sqrt(-np.log(bbw))
        self.t0 = np.sqrt(-np.log(bt)) / self.alpha

    def is_ok(self, bf=0.001):
        return self.fc < self.alpha * np.pi / np.sqrt(-np.log(bf))

    def solve(self, time):
        return np.exp(-(self.alpha ** 2) * (time - self.t0) ** 2) \
               * np.cos(2 * np.pi * self.fc * (time - self.t0))


def animate(i, *fargs):
    fig = fargs[0]
    ax = fargs[1]
    surf = fargs[2]
    grid = fargs[3]
    X = fargs[4]
    Y = fargs[5]

    ax.clear()
    surf = plot_ez(fig, ax, X, Y, grid.data[i])

    return surf,


def plot_ez(fig, ax, X, Y, Z):
    surf_args = {'cmap': cm.coolwarm,
                 'linewidth': 0,
                 'antialiased': False}
    surf = ax.plot_surface(X, Y, Z, **surf_args)
    ax.set_zlim(0, 1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Ez')

    return surf


def plot_time_freq(yt, time, dt, title):
    f = fftpack.fftfreq(len(time), dt)
    F = np.abs(fftpack.fft(yt))
    mask = np.where(f >= 0)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(time, yt)
    axes[0].set_title(title)
    axes[0].set_xlabel('Time (s)')
    axes[0].grid()

    axes[1].loglog(f[mask], F[mask])
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].grid()


def plot_response(input, output, time, dt, title='Response'):
    f = fftpack.fftfreq(len(time), dt)
    Fin = np.abs(fftpack.fft(input))
    Fout = np.abs(fftpack.fft(output))
    threshold = 1e-3 * max(Fin)  # bad results with low Fin
    mask = np.where((f >= 0) & (Fin > threshold))
    response = 20 * np.log10(Fout[mask] / Fin[mask])

    fig, ax = plt.subplots()
    ax.semilogx(f[mask], response, marker='.')
    ax.set_title(title)
    ax.set_ylabel('Response (dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.grid()


def main():
    total_time = 5e-9
    n_frames = 125
    dx = 0.01
    grid = Grid2D(dx, 0.6, 0.6)
    grid.init_pml(8)
    print(grid)
    p1 = np.array([20, 30])
    p2 = np.array([40, 50])
    grid.set_material(p1, p2, er=4.0, cond=0.1)
    src_pos = np.array([[5], [5]])
    source = Gaussian(src_pos, 'Gaussian', 1.0, 20 * grid.dt, 6 * grid.dt)
    # source = Sinusoid(src_pos, 'Sine 1 GHz', 1.0, 1e9)
    # source = SinusoidalGauss(src_pos, 'Sine-Gauss 1 GHz', 5e8, 2e9)
    grid.add_source(source)
    grid.add_probe(Probe(np.array([[50], [50]]), 'Corner'))
    grid.solve(total_time, n_frames)

    X, Y = np.meshgrid(grid.x, grid.y)

    source_data = source.solve(grid.t)
    plot_time_freq(source_data, grid.t, grid.dt, 'Source')
    for probe in grid.probes:
        plot_time_freq(probe.data, grid.t, grid.dt, probe.label)
    plot_response(source_data, probe.data, grid.t, grid.dt, 'Response')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = plot_ez(fig, ax, X, Y, grid.data[0])
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ani = animation.FuncAnimation(
        fig, animate, fargs=(fig, ax, surf, grid, X, Y), interval=40,
        frames=len(grid.data))

    f = r"fdtd-2d.gif"
    print('Saving', f, '...', end='')
    writergif = animation.PillowWriter(fps=25)
    ani.save(f, writer=writergif)
    print(' done.')

    plt.show()


if __name__ == '__main__':
    main()
