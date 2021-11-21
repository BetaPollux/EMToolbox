#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def fdtd_2conductor(L: float, C: float,
                    vs: float, rs: float, rl: float, *,
                    dt: float, dz: float,
                    ndt: int, ndz: int):
    v = np.zeros((ndz + 1, ndt + 1))
    i = np.zeros((ndz, ndt + 1))
    t = np.linspace(0, ndt * dt, ndt + 1)
    z = np.linspace(0, ndz * dz, ndz + 1)
    vsp = 0
    for j in range(ndt):
        v[1:-1, j + 1] = v[1:-1, j] - (dt / dz / C) * (i[1:, j] - i[:-1, j])
        v[0, j + 1] = 1 / (dz / dt * rs * C + 1) * (
            (dz / dt * rs * C - 1) * v[0, j] -
            2 * rs * i[0, j] +
            (vs + vsp)
        )
        vsp = vs
        v[-1, j + 1] = 1 / (dz / dt * rl * C + 1) * (
            (dz / dt * rl * C - 1) * v[-1, j] +
            2 * rl * i[-1, j]
        )
        i[:, j + 1] = i[:, j] - (dt / dz / L) * (v[1:, j + 1] - v[:-1, j + 1])

    return (t, z, v, i)


if __name__ == '__main__':
    print('TLine FDTD')
    # Paul MTL fig 8.23
    zc = 50
    vp = 200e6
    vs = 30
    rs = 0
    rl = 100
    L = zc / vp
    C = 1 / (zc * vp)
    print(L*1e6, 'uH/m', C*1e12, 'pF/m')
    t, z, v, i = fdtd_2conductor(L, C, vs, rs, rl,
                                 dt=10e-9, dz=2.0,
                                 ndt=2000, ndz=200)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(t, v[0, :], label='Src')
    axs[0].plot(t, v[-1, :], label='Load')
    axs[0].set_ylabel('v (V)')
    axs[1].plot(t, i[0, :], label='Src')
    axs[1].plot(t, i[-1, :], label='Load')
    axs[1].set_ylabel('i (A)')
    for ax in axs:
        ax.legend()
        ax.grid()
    axs[-1].set_xlabel('t (s)')
    plt.show()
