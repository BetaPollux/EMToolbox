#!/usr/bin/python3

'''Wrapper interface for tline classes for use by GUI'''

from tline import TLine, TerminatedTLine
import cmath
import numpy as np
import matplotlib.pyplot as plt


def solve(inputs: dict) -> dict:
    zs = float(inputs.get('source_z', 50))
    zl = float(inputs.get('load_z', 50))
    length = float(inputs.get('length', 1))
    tline_r = float(inputs.get('tline_r', 0))
    tline_l = float(inputs.get('tline_l', 50e-6))
    tline_g = float(inputs.get('tline_g', 0))
    tline_c = float(inputs.get('tline_c', 100e-6))
    freq_start = float(inputs.get('freq_start', 100e3))
    freq_stop = float(inputs.get('freq_stop', 1e9))

    tline = TLine(tline_l, tline_c, length, tline_r, tline_g)
    network = TerminatedTLine(tline, zs, zl, 1.0)

    n = 400
    f = np.geomspace(freq_start, freq_stop, n)
    w = 2 * np.pi * f
    src = network.solve(w, 0)
    load = network.solve(w, length)

    fig, ax = plt.subplots()
    ax.semilogx(f, np.abs(src[0]), label='Source')
    ax.semilogx(f, np.abs(load[0]), label='Load')
    ax.set(xlabel='Frequency (Hz)', ylabel='Voltage (V)')
    if max(ax.get_ylim()) < 1:
        ax.set_ylim([0, 1])
    ax.legend()
    ax.grid()

    # plt.ion()
    plt.show()

    calc_f = 10e6
    calc_w = 2 * np.pi * calc_f
    results = {'frequency': f'{calc_f:.3e}',
               'tline_td': f'{tline.delay(calc_w):.3e}',
               'tline_zc': '{:.3f} ohm, {:.3f} rad'.format(
                   *cmath.polar(tline.impedance(calc_w))),
               'tline_vp': f'{tline.velocity(calc_w):.3e}',
               'tline_attn': f"{tline.attn_const(calc_w, units='db'):.3f}",
               'tline_phase': f"{tline.phase_const(calc_w, units='deg'):.3f}"}
    return results
