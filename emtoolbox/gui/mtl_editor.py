#!/usr/bin/python3

'''An editor window for configuring the MTL and cross section'''

import wx
from emtoolbox.gui.helpers import create_choice_field_set, create_text_field_set, layout_fields
from emtoolbox.gui.mtl_canvas import MtlCanvas


class MtlEditor(wx.Dialog):
    '''An editor window for configuring the MTL and cross section'''
    def __init__(self, parent=None, title: str = 'MtlEditor', size=(640, 480)):
        wx.Dialog.__init__(self, parent, title=title, size=size,
                           style=(wx.DEFAULT_DIALOG_STYLE |
                                  wx.RESIZE_BORDER |
                                  wx.MINIMIZE_BOX |
                                  wx.MAXIMIZE_BOX))
        choice_fields = create_choice_field_set(self, self.choice_fields(), wx.CB_READONLY)
        text_fields = create_text_field_set(self, self.input_fields())
        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        self.canvas = MtlCanvas(self, size=(320, 240))

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        field_sizer = layout_fields([*choice_fields, *text_fields])
        content_sizer.Add(field_sizer, 0, 0)
        content_sizer.Add(self.canvas, 1, wx.EXPAND)
        main_sizer.Add(content_sizer, 1, wx.EXPAND)
        main_sizer.Add(btn_sizer, 0, wx.EXPAND)
        self.SetSizer(main_sizer)
        main_sizer.Fit(self)
        main_sizer.SetSizeHints(self)

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
