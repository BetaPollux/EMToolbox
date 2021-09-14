#!/usr/bin/python3

import wx
import emtoolbox.gui.mtl_wrapper as mtl_wrapper
import emtoolbox.gui.helpers as hlp
from emtoolbox.gui.mtl_editor import MtlEditor
from emtoolbox.utils.constants import CHR_OHM


class MtlFrame(wx.Frame):
    '''Top level frame for the multi-conductor transmission line solvers'''
    def __init__(self, parent=None, id=-1,
                 title='Multi-conductor Transmission Lines',
                 pos=wx.DefaultPosition,
                 size=(640, 480)):
        super().__init__(parent, id, title, pos, size)
        self.panel = wx.Panel(self)

        input_fields = hlp.create_text_field_set(self.panel, self.input_fields())
        output_fields = hlp.create_text_field_set(self.panel, self.output_fields(),
                                                  text_style=wx.TE_READONLY)

        solve_btn = wx.Button(self.panel, wx.ID_ANY, 'Solve')
        edit_btn = wx.Button(self.panel, wx.ID_ANY, 'Edit')
        self.Bind(wx.EVT_BUTTON, self.OnSolve, solve_btn)
        self.Bind(wx.EVT_BUTTON, self.OnEdit, edit_btn)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        field_sizer = self.layout_fields(input_fields, output_fields)
        btn_sizer = hlp.layout_buttons([edit_btn, solve_btn])
        main_sizer.Add(field_sizer, 0, wx.EXPAND)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT)
        self.panel.SetSizer(main_sizer)
        main_sizer.Fit(self)
        main_sizer.SetSizeHints(self)

    def layout_fields(self, input_fields: list, output_fields: list) -> wx.Sizer:
        '''Create a sizer containing input and output fields
        Arguments:
            input_fields: [(label, field), ...]
            output_fields: [(label, field), ...]
        Return:
            Sizer containing the fields'''
        # Equalize number of fields per column
        pad = (10, 10)
        num_rows = max(len(input_fields), len(output_fields))
        input_fields.extend([pad] * (num_rows - len(input_fields)))
        output_fields.extend([pad] * (num_rows - len(output_fields)))

        field_sizer = wx.FlexGridSizer(cols=4, vgap=5, hgap=5)
        field_sizer.AddGrowableCol(1, proportion=1)
        field_sizer.AddGrowableCol(3, proportion=1)

        for input_pair, output_pair in zip(input_fields, output_fields):
            in_lbl, in_field = input_pair
            out_lbl, out_field = output_pair
            field_sizer.Add(in_lbl, 0, wx.LEFT, 5)
            field_sizer.Add(in_field, 0, wx.EXPAND)
            field_sizer.Add(out_lbl, 0, wx.LEFT, 5)
            field_sizer.Add(out_field, 0, wx.EXPAND | wx.RIGHT, 5)
        return field_sizer

    def input_fields(self):
        return (hlp.pack_input_params('source_z', f'ZS ({CHR_OHM})', 50),
                hlp.pack_input_params('load_z', f'ZL ({CHR_OHM})', 50),
                hlp.pack_input_params('length', 'Length (m)', 1),
                hlp.pack_input_params('tline_r', f'R ({CHR_OHM}/m)', 2),
                hlp.pack_input_params('tline_l', 'L (H/m)', 500e-9),
                hlp.pack_input_params('tline_g', 'G (S/m)', 1e-8),
                hlp.pack_input_params('tline_c', 'C (F/m)', 100e-12),
                hlp.pack_input_params('freq_start', 'f0 (Hz)', 10e3),
                hlp.pack_input_params('freq_stop', 'f1 (Hz)', 1e9))

    def output_fields(self):
        return (hlp.pack_input_params('frequency', 'At Frequency (Hz)'),
                hlp.pack_input_params('tline_td', 'Td (s)'),
                hlp.pack_input_params('tline_zc', f'Zc ({CHR_OHM})'),
                hlp.pack_input_params('tline_vp', 'Vp (m/s)'),
                hlp.pack_input_params('tline_attn', 'attn (dB/m)'),
                hlp.pack_input_params('tline_phase', 'phase (deg/m)'))

    def OnSolve(self, event):
        inputs = hlp.parse_input_fields(self, self.input_fields())
        outputs = mtl_wrapper.solve(inputs, parent_window=self)
        hlp.populate_output_fields(self, outputs)

    def OnEdit(self, event):
        editor = MtlEditor()
        resp = editor.ShowModal()
        if resp == wx.ID_OK:
            line = editor.get_mtl()
            print(f'WireMtl # wires {len(line.wires)}, type {type(line.ref)}')
            print(f'L\n{line.inductance()}')
            print(f'C\n{line.capacitance()}')
            outputs = {
                'tline_l': f'{float(line.inductance()):.3e}',
                'tline_c': f'{float(line.capacitance()):.3e}',
                'tline_r': 0.0,
                'tline_g': 0.0
            }
            hlp.populate_output_fields(self, outputs)
        editor.Destroy()


class MtlApp(wx.App):
    '''Top level application for the multi-conductor
    transmission line solvers'''
    def OnInit(self):
        '''Set up top level frames'''
        self.frame = MtlFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True


def main():
    app = MtlApp(redirect=False)
    app.MainLoop()


if __name__ == '__main__':
    main()
