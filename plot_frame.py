#!/usr/bin/python3

import wx
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_wxagg \
    import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg \
    import NavigationToolbar2WxAgg as NavigationToolbar


class PlotTab(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.figure = mpl.figure.Figure(figsize=(5, 2))
        self.axes = [self.figure.gca()]

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

    def set_axis(self, xlabel: str, ylabel: str, ax: int = 0):
        self.axes[ax].set_xlabel(xlabel)
        self.axes[ax].set_ylabel(ylabel)
        self.axes[ax].grid()

    def plot(self, x, y, ax: int = 0):
        self.axes[ax].plot(x, y)


class PlotFrame(wx.Frame):
    def __init__(self, title: str = 'PlotFrame'):
        wx.Frame.__init__(self, None, title=title, size=(640, 480))

        self.panel = wx.Panel(self)
        self.notebook = wx.Notebook(self.panel)
        self.pages = []

        sizer = wx.BoxSizer()
        sizer.Add(self.notebook, 1, wx.EXPAND)
        self.panel.SetSizer(sizer)

    def add_page(self, title: str):
        page = PlotTab(self.notebook)
        self.pages.append(page)
        self.notebook.AddPage(page, title)
        return page


if __name__ == "__main__":
    app = wx.App(redirect=False)
    frame = PlotFrame()

    x = np.linspace(0, 100, 50)
    v = 1 / x**2
    i = 1 + x

    voltages = frame.add_page('Voltages')
    voltages.set_axis('x (m)', 'Voltage (V)')
    voltages.plot(x, v)
    currents = frame.add_page('Currents')
    currents.set_axis('t (s)', 'Current (A)')
    currents.plot(x, i)
    frame.Show()
    app.MainLoop()
