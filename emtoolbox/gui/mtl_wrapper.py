#!/usr/bin/python3

'''Wrapper interface for tline classes for use by GUI'''

import cmath
import numpy as np
from emtoolbox.tline.tline import TLine, TerminatedTLine
from emtoolbox.gui.plot_frame import PlotFrame
from emtoolbox.utils.constants import CHR_OHM

# TODO replace with an MtlPlotter class
# TODO create new MtlNetworkSolver class


def solve(inputs: dict, parent_window=None) -> dict:
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
    src = np.abs(network.solve(w, 0))
    load = np.abs(network.solve(w, length))
    zc = np.abs(tline.impedance(w))
    attn = tline.attn_const(w)
    velocity = tline.velocity(w)

    pages = (('Voltage', 'Voltage (V)',
              (src[0], 'Source'), (load[0], 'Load')),
             ('Current', 'Current (A)',
              (src[1], 'Source'), (load[1], 'Load')),
             ('Characteristic Impedance', f'Magnitude of Zc ({CHR_OHM})',
              (zc, '|Zc|')),
             ('Attenuation', 'Attenuation (Np/m)',
              (attn, r'$\alpha$')),
             ('Velocity', 'Velocity (m/s)',
              (velocity, 'Vp')))
    frame = PlotFrame(parent=parent_window)
    for title, units, *curves in pages:
        page = frame.add_page(title)
        page.set_axis('Frequency (Hz)', units, xscale='log')
        for y_data, label in curves:
            page.plot(f, y_data, label=label)
        page.set_legend()
        page.set_grid()
    frame.Show()

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
