#!/usr/bin/python3

import sys
import wx
import mtl_wrapper


class MtlFrame(wx.Frame):
    '''Top level frame for the multi-conductor transmission line solvers'''
    def __init__(self, parent=None, id=-1,
                 title='Multi-conductor Transmission Lines',
                 pos=wx.DefaultPosition,
                 size=(640, 480)):
        super().__init__(parent, id, title, pos, size)
        self.panel = wx.Panel(self)

        input_fields = self.create_text_field_set(self.input_fields())
        output_fields = self.create_text_field_set(self.output_fields(),
                                                   wx.TE_READONLY)

        solve_btn = wx.Button(self.panel, wx.ID_ANY, 'Solve')
        self.Bind(wx.EVT_BUTTON, self.OnSolve, solve_btn)

        fields_sizer = wx.BoxSizer(wx.HORIZONTAL)
        fields_sizer.Add(input_fields, 1, wx.ALL, 5)
        fields_sizer.Add(output_fields, 1, wx.ALL, 5)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(fields_sizer, 0, wx.EXPAND)
        main_sizer.Add(solve_btn, 0, wx.ALIGN_CENTER)
        self.panel.SetSizer(main_sizer)

    def parse_input_fields(self):
        result = {}
        for name, *_ in self.input_fields():
            field = self.FindWindowByName(name)
            result[name] = field.GetValue()
        return result

    def populate_output_fields(self, fields):
        for name, value in fields.items():
            field = self.FindWindowByName(name)
            if field:
                field.SetValue(str(value))
            else:
                print(f'Could not populate output field: {name}, {value}',
                      file=sys.stderr)

    def input_fields(self):
        return (('source_z', 'ZS (ohms)', 50),
                ('load_z', 'ZL (ohms)', 50),
                ('length', 'Length (m)', 1),
                ('tline_r', 'R (ohms)', 2),
                ('tline_l', 'L (henries)', 500e-9),
                ('tline_g', 'G (siemens)', 1e-8),
                ('tline_c', 'C (farads)', 100e-12),
                ('freq_start', 'f0 (Hz)', 10e3),
                ('freq_stop', 'f1 (Hz)', 1e9))

    def output_fields(self):
        return (('frequency', 'At Frequency (Hz)', ''),
                ('tline_td', 'TD (s)', ''),
                ('tline_zc', 'ZC (ohms)', ''),
                ('tline_vp', 'vp (m/s)', ''),
                ('tline_attn', 'attn (dB/m)', ''),
                ('tline_phase', 'phase (deg/m)', ''))

    def create_text_field_set(self, fields, text_style=0):
        text_field_sizer = wx.FlexGridSizer(cols=2, vgap=5, hgap=5)
        text_field_sizer.AddGrowableCol(1, 1)
        for name, label, default in fields:
            static, text = self.create_text_field(self.panel,
                                                  name, label, default,
                                                  text_style)
            text_field_sizer.Add(static, 0, wx.ALIGN_CENTER_VERTICAL)
            text_field_sizer.Add(text, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
        return text_field_sizer

    def create_text_field(self, parent, name, label,
                          default='', text_style=0):
        '''Returns a (StaticText, TextCtrl)'''
        static = wx.StaticText(parent, wx.ID_ANY, label=label)
        text = wx.TextCtrl(parent, wx.ID_ANY,
                           value=str(default), name=name,
                           style=text_style)
        return (static, text)

    def OnSolve(self, event):
        outputs = mtl_wrapper.solve(self.parse_input_fields())
        self.populate_output_fields(outputs)


class MtlApp(wx.App):
    '''Top level application for the multi-conductor
    transmission line solvers'''
    def OnInit(self):
        '''Set up top level frames'''
        self.frame = MtlFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True


if __name__ == '__main__':
    app = MtlApp(redirect=False)
    app.MainLoop()
