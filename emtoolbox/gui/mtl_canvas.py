#!/usr/bin/python3

'''A canvas window for drawing transmission lines'''

import wx
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg \
    import NavigationToolbar2WxAgg as NavigationToolbar
from emtoolbox.tline.wire import Wire, Plane, Shield
from emtoolbox.utils.constants import RGB_CU, RGB_DIEL

BG_COLOR = '#e1e1a1'


class MtlCanvas(wx.Window):
    def __init__(self, parent, size=(80, 80)):
        wx.Window.__init__(self, parent, size=size)
        self.figure = mpl.figure.Figure(figsize=(5, 2))
        self.ax = self.figure.gca()
        self.ax.set_facecolor(BG_COLOR)

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

    def add_conductor(self, conductor):
        if type(conductor) == Wire:
            x, y, r = conductor.x, conductor.y, conductor.radius
            shapes = [mpl.patches.Circle([x, y], r, color=RGB_CU, fill=True)]
        elif type(conductor) == Plane:
            w, t = conductor.width, conductor.thickness
            shapes = [mpl.patches.Rectangle([-0.5 * w, -t], w, t, color=RGB_CU, fill=True)]
        elif type(conductor) == Shield:
            r, t = conductor.radius, conductor.thickness
            shapes = [mpl.patches.Circle([0, 0], r + t, color=RGB_CU, fill=True),
                      mpl.patches.Circle([0, 0], r, color=BG_COLOR, fill=True)]
        for shape in shapes:
            self.ax.add_patch(shape)
        self.ax.autoscale_view()
        self.ax.set_aspect(1)

    def clear_shapes(self):
        self.ax.cla()

    def redraw(self):
        self.figure.canvas.draw()


if __name__ == '__main__':
    class MtlCanvasFrame(wx.Frame):
        def __init__(self, parent):
            wx.Frame.__init__(self, parent, -1, "Mtl Canvas Frame",
                              size=(800, 600))
            self.canvas = MtlCanvas(self)

    app = wx.App(redirect=False)
    frame = MtlCanvasFrame(None)
    frame.canvas.add_conductor(Plane())
    frame.canvas.add_conductor(Wire(0, 0, 2.5))
    frame.canvas.add_conductor(Wire(-0.75, 0, 0.5))
    frame.canvas.add_conductor(Wire(0.75, 0.25, 0.5))
    frame.Show(True)
    app.MainLoop()
