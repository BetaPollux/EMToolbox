#!/usr/bin/python3

'''An editor window for configuring the MTL and cross section'''

import wx
import emtoolbox.gui.helpers as hlp
from emtoolbox.gui.mtl_canvas import MtlCanvas
import emtoolbox.tline.coax as coax


class MtlEditor(wx.Dialog):
    '''An editor window for configuring the MTL and cross section'''
    def __init__(self, parent=None, title: str = 'MtlEditor', size=(640, 480)):
        wx.Dialog.__init__(self, parent, title=title, size=size,
                           style=(wx.DEFAULT_DIALOG_STYLE |
                                  wx.RESIZE_BORDER |
                                  wx.MINIMIZE_BOX |
                                  wx.MAXIMIZE_BOX))
        choice_fields = hlp.create_choice_field_set(self, self.choice_fields(), wx.CB_READONLY)
        text_fields = hlp.create_text_field_set(self, self.input_fields())
        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        self.canvas = MtlCanvas(self, size=(320, 240))

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        field_sizer = hlp.layout_fields([*choice_fields, *text_fields])
        content_sizer.Add(field_sizer, 0, 0)
        content_sizer.Add(self.canvas, 1, wx.EXPAND)
        main_sizer.Add(content_sizer, 1, wx.EXPAND)
        main_sizer.Add(btn_sizer, 0, wx.EXPAND)
        self.SetSizer(main_sizer)
        main_sizer.Fit(self)
        main_sizer.SetSizeHints(self)

    def parse_rlgc(self) -> dict:
        '''Parse selections and return the RLGC calculation functions'''
        result = {}
        result['L'] = coax.inductance
        result['C'] = coax.capacitance
        for choice in self.choice_fields():
            if choice['name'] == 'r_models':
                sel = self.FindWindowByName(choice['name']).GetSelection()
                result['R'] = choice['functions'][sel]
            elif choice['name'] == 'g_models':
                sel = self.FindWindowByName(choice['name']).GetSelection()
                result['G'] = choice['functions'][sel]
        return result

    def choice_fields(self):
        return ({'name':      'geometry',
                 'label':     'Geometry',
                 'choices':   ['coax', 'two-wire', 'wire-over-ground',
                              'microstrip', 'stripline']},
                {'name':      'r_models',
                 'label':     'Resistance Model',
                 'choices':   ['None', 'DC', 'Skin Effect'],
                 'functions': [None, coax.resistance_dc, coax.resistance_skin_effect]},
                {'name':      'g_models',
                 'label':     'Dielectric Model',
                 'choices':   ['None', 'Conductivity', 'Loss Tangent'],
                 'functions': [None, coax.conductance_simple, coax.conductance_loss_tangent]})

    def input_fields(self):
        return ({'name':    'radius_w',
                 'label':   'Inner radius',
                 'default': 1e-3},
                {'name':    'radius_s',
                 'label':   'Outer radius',
                 'default': 3e-3},
                {'name':    'epsr',
                 'label':   'Rel. Permittivity',
                 'default': 1.8})


if __name__ == '__main__':
    app = wx.App(redirect=False)
    dialog = MtlEditor()
    dialog.ShowModal()
    app.MainLoop()
