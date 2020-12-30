#!/usr/bin/python3

import wx
import emtoolbox.gui.mtl_wrapper as mtl_wrapper
from emtoolbox.gui.helpers import create_text_field_set, populate_output_fields, parse_input_fields
from emtoolbox.gui.mtl_editor import MtlEditor

class MtlFrame(wx.Frame):
    '''Top level frame for the multi-conductor transmission line solvers'''
    def __init__(self, parent=None, id=-1,
                 title='Multi-conductor Transmission Lines',
                 pos=wx.DefaultPosition,
                 size=(640, 480)):
        super().__init__(parent, id, title, pos, size)
        self.panel = wx.Panel(self)

        input_fields = create_text_field_set(self.panel, self.input_fields())
        output_fields = create_text_field_set(self.panel, self.output_fields(),
                                              text_style=wx.TE_READONLY)

        solve_btn = wx.Button(self.panel, wx.ID_ANY, 'Solve')
        edit_btn = wx.Button(self.panel, wx.ID_ANY, 'Edit')
        self.Bind(wx.EVT_BUTTON, self.OnSolve, solve_btn)
        self.Bind(wx.EVT_BUTTON, self.OnEdit, edit_btn)

        fields_sizer = wx.BoxSizer(wx.HORIZONTAL)
        fields_sizer.Add(input_fields, 1, wx.ALL, 5)
        fields_sizer.Add(output_fields, 1, wx.ALL, 5)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(fields_sizer, 0, wx.EXPAND)
        main_sizer.Add(edit_btn, 0, wx.ALIGN_CENTER)
        main_sizer.Add(solve_btn, 0, wx.ALIGN_CENTER)
        self.panel.SetSizer(main_sizer)

    def input_fields(self):
        return (('source_z', f'ZS ({chr(0x3a9)})', 50),
                ('load_z', f'ZL ({chr(0x3a9)})', 50),
                ('length', 'Length (m)', 1),
                ('tline_r', f'R ({chr(0x3a9)}/m)', 2),
                ('tline_l', 'L (H/m)', 500e-9),
                ('tline_g', 'G (S/m)', 1e-8),
                ('tline_c', 'C (F/m)', 100e-12),
                ('freq_start', 'f0 (Hz)', 10e3),
                ('freq_stop', 'f1 (Hz)', 1e9))

    def output_fields(self):
        return (('frequency', 'At Frequency (Hz)', ''),
                ('tline_td', 'Td (s)', ''),
                ('tline_zc', f'Zc ({chr(0x3a9)})', ''),
                ('tline_vp', 'Vp (m/s)', ''),
                ('tline_attn', 'attn (dB/m)', ''),
                ('tline_phase', 'phase (deg/m)', ''))

    def OnSolve(self, event):
        inputs = parse_input_fields(self, self.input_fields())
        outputs = mtl_wrapper.solve(inputs, parent_window=self)
        populate_output_fields(self, outputs)

    def OnEdit(self, event):
        editor = MtlEditor()
        resp = editor.ShowModal()
        print(resp)


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
