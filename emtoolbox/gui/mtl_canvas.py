#!/usr/bin/python3

'''A canvas window for drawing transmission lines'''

import wx
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg \
    import NavigationToolbar2WxAgg as NavigationToolbar

class MtlCanvas(wx.Window):
    def __init__(self, parent, size=(80, 80)):
        wx.Window.__init__(self, parent, size=size)
        self.figure = mpl.figure.Figure(figsize=(5, 2))
        self.ax = self.figure.gca()

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

    def add_shape(self, style, *args):
        if style == 'circle':
            x, y, r = args
            shape = mpl.patches.Circle([x, y], r, fill=False)
        elif style == 'rect':
            x, y, w, h = args
            shape = mpl.patches.Rectangle([x, y], w, h, fill=False)
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
                    size=(800,600))
            self.canvas = MtlCanvas(self)

    app = wx.App(redirect=False)
    frame = MtlCanvasFrame(None)
    frame.canvas.add_shape('rect', 0, 1.0, 0.5, 0.75)
    frame.canvas.add_shape('circle', 0, 0, 2.5)
    frame.canvas.add_shape('circle', -0.75, 0, 0.5)
    frame.canvas.add_shape('circle', 0.75, 0.25, 0.5)
    frame.Show(True)
    app.MainLoop()
