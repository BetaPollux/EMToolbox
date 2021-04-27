#!/usr/bin/python3

import numpy as np
from emtoolbox.gui.plot_frame import PlotFrame
from emtoolbox.utils.constants import CHR_OHM


class MtlPlotter():
    def __init__(self, tline, parent_window=None):
        self.tline = tline
        self.plot_frame = PlotFrame(parent=parent_window)
        self.init_plot_frame(self.tline, self.plot_frame)
    
    def init_plot_frame(self, tline, frame):
        f = np.logspace(2, 9, 201)
        zc = tline.char_impedance() * np.ones_like(f)
        attn = tline.attn_const(f) * np.ones_like(f)
        velocity = tline.velocity() * np.ones_like(f)
        pages = (
            ('Characteristic Impedance', f'|Zc| ({CHR_OHM})',
                (zc, '|Zc|')),
            ('Attenuation', 'Attenuation (Np/m)',
                (attn, r'$\alpha$')),
            ('Velocity', 'Velocity (m/s)',
                (velocity, 'Vp'))
        )

        for title, units, *curves in pages:
            page = frame.add_page(title)
            page.set_axis('Frequency (Hz)', units, xscale='log')
            for y_data, label in curves:
                page.plot(f, y_data, label=label)
            page.set_legend()
            page.set_grid()
    
    def Show(self, show=True):
        self.plot_frame.Show(show)

