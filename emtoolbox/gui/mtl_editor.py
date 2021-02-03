#!/usr/bin/python3

'''An editor window for configuring the MTL and cross section'''

import wx
import emtoolbox.gui.helpers as hlp
from emtoolbox.gui.mtl_canvas import MtlCanvas
import emtoolbox.tline.coax as coax
import emtoolbox.utils.constants as const
from emtoolbox.tline.tline import TLine


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

        for _, txt in text_fields:
            self.Bind(wx.EVT_TEXT, self.OnInput, txt)

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
        self.update_canvas()

    def OnInput(self, event):
        self.update_canvas()

    def update_canvas(self) -> None:
        inputs = hlp.parse_input_fields(self, 
                                        [{'name': name} for name in ('radius_w', 'radius_s')],
                                        float)
        rw = inputs.get('radius_w', 1e-3)
        rs = inputs.get('radius_s', 3e-3)
        self.canvas.clear_shapes()
        self.canvas.add_shape('circle', 0, 0, rs)
        self.canvas.add_shape('circle', 0, 0, rw)
        self.canvas.redraw()

    def parse_rlgc_functions(self) -> dict:
        '''Parse selections and return the RLGC calculation functions'''
        result = {}
        result['l'] = coax.inductance
        result['c'] = coax.capacitance
        for choice in self.choice_fields():
            if choice['name'] == 'r_models':
                sel = self.FindWindowByName(choice['name']).GetSelection()
                result['r'] = choice['functions'][sel]
            elif choice['name'] == 'g_models':
                sel = self.FindWindowByName(choice['name']).GetSelection()
                result['g'] = choice['functions'][sel]
        return result

    def parse_rlgc(self) -> dict:
        '''Collect the RLGC generator functions'''
        functions = self.parse_rlgc_functions()
        inputs = hlp.parse_input_fields(self, self.input_fields(), float)
        result = {}
        for key, func in functions.items():
            if func:
                result[key] = func(**inputs)  # TODO filter the kwargs
        return result

    def parse_length(self) -> float:
        '''Get the line length'''    
        return hlp.parse_input_fields(self, [{'name': 'length'}], float)['length']

    def get_tline(self) -> TLine:
        '''Get the transmission line'''
        line = TLine(**self.parse_rlgc(), length=self.parse_length())
        return line

    def choice_fields(self) -> list:
        '''Collection of drop-down choices in the editor'''
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

    def input_fields(self) -> list:
        '''Collection of user text input fields'''
        return (hlp.pack_input_params('radius_w', 'Inner radius (m)', 1e-3),
                hlp.pack_input_params('radius_s', 'Outer radius (m)', 3e-3),
                hlp.pack_input_params('epsr', 'Rel. Permittivity', 1.8),
                hlp.pack_input_params('length', 'Length (m)', 1.0),
                hlp.pack_input_params('cond_c', 'Wire Conductivity (s/m)', const.COND_CU),
                hlp.pack_input_params('cond_d', 'Dielectric Conductivity (s/m)', 1e-14),
                hlp.pack_input_params('loss_tangent', 'Loss Tangent', 0.02))


if __name__ == '__main__':
    app = wx.App(redirect=False)
    dialog = MtlEditor()
    dialog.ShowModal()
    app.MainLoop()
