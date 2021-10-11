#!/usr/bin/python3

'''Wrapper interface for tline classes for use by GUI'''

import cmath
import numpy as np
from emtoolbox.tline.tline import TLine
from emtoolbox.tline.mtl_network import MtlNetwork
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

    # TODO only 1 frequency due to MtlNetwork.solve implementation
    # n = 400
    # f = np.geomspace(freq_start, freq_stop, n)
    f = freq_start

    tline = TLine(tline_l, tline_c, freq=f, length=length, R=tline_r, G=tline_g)
    network = MtlNetwork(tline, zs, zl)
    src = np.abs(network.solve(0))
    load = np.abs(network.solve(length))
    zc = np.abs(tline.char_impedance())
    attn = tline.attn_const()
    velocity = tline.velocity()

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
            page.plot(f, y_data, label=label, marker='x')
        page.set_legend()
        page.set_grid()
    frame.Show()

    # TODO will need update for multiple frequencies
    calc_f = f
    results = {'frequency': f'{calc_f:.3e}',
               'tline_td': f'{tline.delay():.3e}',
               'tline_zc': '{:.3f} ohm, {:.3f} rad'.format(
                   *cmath.polar(tline.char_impedance())),
               'tline_vp': f'{tline.velocity():.3e}',
               'tline_attn': f"{tline.attn_const(units='db'):.3f}",
               'tline_phase': f"{tline.phase_const(units='deg'):.3f}"}
    return results
