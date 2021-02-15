#! /usr/bin/python3

from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as collections
from scipy import fftpack

eps0 = 8.8541878176e-12


class Grid1D:
    """FDTD 1D Grid, TEM propagating along x-axis"""
    def __init__(self, dx, width):
        self.dx = dx
        self.x = np.arange(0.0, width, dx)
        self.ndx = len(self.x)
        self.dt = self.dx / (2 * 3e8)
        self.ez = np.zeros(self.ndx)
        self.hy = np.zeros(self.ndx)
        self.ca = np.ones(self.ndx)
        self.cb = 0.5 * np.ones(self.ndx)
        self.sources = []
        self.probes = []
        self.data = []

    def __repr__(self):
        s = 'ndx {0:}\n'.format(self.ndx) \
            + 'x   {0:.3e}   m\n'.format(max(self.x)) \
            + 'dx  {0:.3e}   m\n'.format(self.dx) \
            + 'dt  {0:.3e}   s'.format(self.dt)
        return s

    def set_material(self, start, stop, er=1.0, cond=0.0):
        indices = (self.x >= start) & (self.x <= stop)
        eaf = self.dt * cond / (2 * eps0 * er)
        self.ca[indices] = (1 - eaf) / (1 + eaf)
        self.cb[indices] = 0.5 / (er * (1 + eaf))

    def solve(self, total_time, n_frames=125):
        start_time = datetime.now()
        self.t = np.arange(total_time, step=self.dt)
        frame_ids = np.linspace(0, len(self.t) - 1,
                                int(n_frames), dtype='int64')
        for probe in self.probes:
            probe.data = np.zeros(len(self.t))

        abc_left = [0, 0]
        abc_right = [0, 0]

        print('Solving {0} time steps'.format(len(self.t)))
        print_ids = np.linspace(0, len(self.t) - 1, 11, dtype='int64')

        for time_id, time in enumerate(self.t):
            if time_id in print_ids:
                print('Step {0} {1:.0f}%'.format(
                    time_id,
                    100.0 * time_id / len(self.t)))

            self.ez[1:] = self.ca[1:] * self.ez[1:] + self.cb[1:] * (self.hy[:-1] - self.hy[1:])

            for source in self.sources:
                self.ez[source.position] += source.solve(time)

            self.ez[0] = abc_left.pop()
            abc_left.insert(0, self.ez[1])
            self.ez[-1] = abc_right.pop()
            abc_right.insert(0, self.ez[-2])

            self.hy[:-1] = self.hy[:-1] + 0.5 * (self.ez[:-1] - self.ez[1:])

            for probe in self.probes:
                probe.data[time_id] = self.ez[probe.position]

            if time_id in frame_ids:
                self.data.append(self.ez.copy())

        end_time = datetime.now()
        print('Solve complete!')
        print(f'Elapsed {end_time - start_time}')

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
    line = fargs[0]
    data = fargs[1]
    line.set_ydata(data[i])
    return line,


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
    threshold = 1e-6 * max(Fin)  # bad results with low Fin
    mask = np.where((f >= 0) & (Fin > threshold))
    response = 20 * np.log10(Fout[mask] / Fin[mask])

    fig, ax = plt.subplots()
    ax.semilogx(f[mask], response, marker='.')
    ax.set_title(title)
    ax.set_ylabel('Response (dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.grid()


def create_shield(thk, er, cond):
    dx = thk / 10
    width = 20 * thk
    grid = Grid1D(dx, width)
    pos = 0.5 * width
    grid.set_material(pos, pos + thk, er=er, cond=cond)
    return grid


def main():
    total_time = 20e-9
    n_frames = 250
    dx = 0.01
    grid = Grid1D(dx, 2.0)
    grid.set_material(1.0, 1.5, er=4.0, cond=0.04)
    # grid = create_shield(0.25e-3, 1.0, 11.6e8)
    print(grid)

    src_z = 5
    source = Gaussian(src_z, 'Gaussian', 1.0, 40 * grid.dt, 12 * grid.dt)
    # source = Sinusoid(src_z, 'Sine', 1.0, 200e6)
    # source = SinusoidalGauss(src_z, 'Sine-Gauss', 1e7, 1e9)
    grid.add_source(source)
    grid.add_probe(Probe(grid.ndx - 5, 'Transmitted'))
    grid.solve(total_time, n_frames)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(grid.x, grid.ez)
    axes[0].set_ylabel('Ez')
    axes[1].plot(grid.x, grid.hy)
    axes[1].set_ylabel('Hy')
    axes[1].set_xlabel('x (m)')

    fig2, ax2 = plt.subplots()
    ax2.set_ylim(-2, 2)
    eline, = ax2.plot(grid.x, grid.ez)
    ax2.set_ylabel('Ez')
    ax2.set_xlabel('x (m)')
    collection = collections.BrokenBarHCollection.span_where(
        grid.x, ymin=-2, ymax=2, where=grid.cb < 0.5,
        facecolor='red', alpha=0.5)
    ax2.add_collection(collection)
    ani = animation.FuncAnimation(
        fig2, animate, fargs=(eline, grid.data), interval=40,
        blit=True, frames=len(grid.data))

    f = r"fdtd-1d.gif"
    print('Saving', f, '...', end='')
    writergif = animation.PillowWriter(fps=25)
    ani.save(f, writer=writergif)
    print(' done.')

    source_data = source.solve(grid.t)
    plot_time_freq(source_data, grid.t, grid.dt, 'Source')
    for probe in grid.probes:
        plot_time_freq(probe.data, grid.t, grid.dt, probe.label)
    plot_response(source_data, probe.data, grid.t, grid.dt, 'Response')
    plt.show()


if __name__ == '__main__':
    main()
