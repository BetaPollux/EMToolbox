#!/usr/bin/python3

'''An editor window for configuring the MTL and cross section'''

import wx
from emtoolbox.gui.helpers import create_choice_field_set, create_text_field_set


class MtlEditor(wx.Dialog):
    '''An editor window for configuring the MTL and cross section'''
    def __init__(self, parent=None, title: str = 'MtlEditor'):
        wx.Dialog.__init__(self, parent, title=title)
        choice_sizer = create_choice_field_set(self, self.choice_fields(), wx.CB_READONLY)
        text_sizer = create_text_field_set(self, self.input_fields())

        ok_btn = wx.Button(self, wx.ID_OK, 'Ok')
        cancel_btn = wx.Button(self, wx.ID_CANCEL, 'Cancel')
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(ok_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(choice_sizer, 0, wx.ALL, 5)
        main_sizer.Add(text_sizer, 0, wx.ALL, 5)
        main_sizer.Add(btn_sizer)
        self.SetSizer(main_sizer)
        self.Fit()

    def choice_fields(self):
        return (('geometry', 'Geometry', ['coax', 'two-wire', 'wire-over-ground',
                                          'microstrip', 'stripline']),
                ('r_models', 'Resistance Model', ['None', 'DC', 'Skin Effect']),
                ('g_models', 'Dielectric Model', ['None', 'Conductivity', 'Loss Tangent']))

    def input_fields(self):
        return (('radius_w', 'Inner radius', 1e-3),
                ('radius_s', 'Outer radius', 3e-3),
                ('epsr', 'Rel. Permittivity', 1.8))


if __name__ == '__main__':
    app = wx.App(redirect=False)
    dialog = MtlEditor()
    dialog.ShowModal()
    app.MainLoop()
