#!/usr/bin/python3

'''An editor window for configuring the MTL and cross section'''

import wx
import wx.grid
import emtoolbox.gui.helpers as hlp
from emtoolbox.gui.mtl_canvas import MtlCanvas
from emtoolbox.gui.mtl_plotter import MtlPlotter
import emtoolbox.utils.constants as const
from emtoolbox.tline.wire_mtl import WireMtl
from emtoolbox.tline.wire import Wire, Plane, Shield
from emtoolbox.tline.lossless_mtl import LosslessMtl


DEFAULT_RS = 3e-3
DEFAULT_RW = 0.5e-3

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
        self.canvas = MtlCanvas(self, size=(640, 480))

        self.grid = wx.grid.Grid(self)
        self.grid.CreateGrid(10, 3)
        col_labels = ['x', 'y','radius']
        for i, lbl in enumerate(col_labels):
            self.grid.SetColLabelValue(i, lbl)

        plot_btn = wx.Button(self, wx.ID_ANY, 'Plot')
        self.Bind(wx.EVT_BUTTON, self.OnPlot, plot_btn)

        for _, txt in text_fields:
            self.Bind(wx.EVT_TEXT, self.OnInput, txt)
        for _, cmb in choice_fields:
            self.Bind(wx.EVT_COMBOBOX, self.OnInput, cmb)
        self.Bind(wx.grid.EVT_GRID_CELL_CHANGED, self.OnInput, self.grid)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        field_sizer = wx.BoxSizer(wx.VERTICAL)
        field_sizer.Add(hlp.layout_fields([*choice_fields, *text_fields]), 0, 0)
        field_sizer.Add(plot_btn, 0, 0)
        field_sizer.Add(self.grid, 0, 0)
        content_sizer.Add(field_sizer, 0, 0)
        content_sizer.Add(self.canvas, 1, wx.EXPAND)
        main_sizer.Add(content_sizer, 1, wx.EXPAND)
        main_sizer.Add(btn_sizer, 0, wx.EXPAND)
        self.SetSizer(main_sizer)
        main_sizer.Fit(self)
        main_sizer.SetSizeHints(self)

        self.wires = [Wire(-2 * DEFAULT_RW, 0, DEFAULT_RW),
                      Wire(2 * DEFAULT_RW, 0, DEFAULT_RW)]
        self.update_grid(self.wires)
        self.update_canvas(self.get_mtl())

    def OnInput(self, event):
        try:
            self.update_canvas(self.get_mtl())
        except ValueError:
            pass
    
    def OnPlot(self, event):
        mtl = LosslessMtl(self.get_mtl())
        plotter = MtlPlotter(mtl, parent_window=self)
        plotter.Show()

    def update_grid(self, wires):
        for i, w in enumerate(wires):
            self.grid.SetCellValue(i, 0, str(w.x))
            self.grid.SetCellValue(i, 1, str(w.y))
            self.grid.SetCellValue(i, 2, str(w.radius))

    def parse_grid(self):
        wires = []
        try:
            for row in range(self.grid.GetNumberRows()):
                x = float(self.grid.GetCellValue(row, 0))
                y = float(self.grid.GetCellValue(row, 1))
                r = float(self.grid.GetCellValue(row, 2))
                wires.append(Wire(x, y, r))
        except ValueError:
            pass
        return wires

    def parse_reference(self):
        inputs = hlp.parse_input_fields(self, [{'name': 'radius_s'}], float)
        rs = inputs.get('radius_s', DEFAULT_RS)
        sel = self.FindWindowByName('reference').GetStringSelection()
        if sel == 'wire':
            return None
        elif sel == 'plane':
            return Plane(2 * rs)
        elif sel == 'shield':
            return Shield(rs)
        else:
            raise Exception('Unrecognized reference selection')

    def update_canvas(self, mtl) -> None:
        self.canvas.clear_shapes()
        self.canvas.add_conductor(mtl.ref)
        for w in mtl.wires:
            self.canvas.add_conductor(w)
        self.canvas.redraw()

    def get_mtl(self) -> WireMtl:
        inputs = hlp.parse_input_fields(self, [{'name': 'epsr'}], float)
        epsr = inputs.get('epsr', 1.0)
        ref = self.parse_reference()
        if ref is None:
            all_wires = self.parse_grid()
            ref = all_wires[0]
            wires = all_wires[1:]
        else:
            wires = self.parse_grid()
        return WireMtl(wires, ref, epsr)

    def choice_fields(self) -> list:
        '''Collection of drop-down choices in the editor'''
        return ({'name':      'reference',
                 'label':     'Reference',
                 'choices':   ['wire', 'plane', 'shield']},
                {'name':      'r_models',
                 'label':     'Resistance Model',
                 'choices':   ['None', 'DC', 'Skin Effect']},
                {'name':      'g_models',
                 'label':     'Dielectric Model',
                 'choices':   ['None', 'Conductivity', 'Loss Tangent']})

    def input_fields(self) -> list:
        '''Collection of user text input fields'''
        return (hlp.pack_input_params('radius_s', 'Shield radius (m)', DEFAULT_RS),
                hlp.pack_input_params('epsr', 'Rel. Permittivity', 1.0),
                hlp.pack_input_params('cond_c', 'Wire Conductivity (s/m)', const.COND_CU),
                hlp.pack_input_params('cond_d', 'Dielectric Conductivity (s/m)', 1e-14),
                hlp.pack_input_params('loss_tangent', 'Loss Tangent', 0.02))


if __name__ == '__main__':
    app = wx.App(redirect=False)
    dialog = MtlEditor()
    dialog.ShowModal()
    app.MainLoop()
