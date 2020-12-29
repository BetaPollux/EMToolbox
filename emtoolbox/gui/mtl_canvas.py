#!/usr/bin/python3

'''A canvas window for drawing transmission lines'''

import wx

class MtlCanvas(wx.Window):
    def __init__(self, parent, ID):
        wx.Window.__init__(self, parent, ID)
        self.buffer = None
        self.reInitBuffer = True
        self.scale = 1.0
        self.originalSize = None
        self.shapes = []
        self.SetBackgroundColour("White")
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def InitBuffer(self):
        size = self.GetClientSize()
        self.buffer = wx.Bitmap(size.width, size.height)
        dc = wx.BufferedDC(None, self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()

    def DrawShapes(self):
        if self.originalSize:
            dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
            x_orig, y_orig = self.originalSize
            x_new, y_new = dc.GetSize()
            x_scale = x_new / x_orig
            y_scale = y_new / y_orig
            self.scale = min(x_scale, y_scale)
            dc.SetUserScale(self.scale, self.scale)
            print('size', dc.GetSize(), 'scale', self.scale)
            for style, *params in self.shapes:
                if style == 'circle':
                    dc.DrawCircle(*params)
                elif style == 'rect':
                    dc.DrawRectangle(*params)
            self.reInitBuffer = False

    def AddShape(self, style, *args):
        self.shapes.append((style, *args))

    def OnSize(self, event):
        self.reInitBuffer = True

    def OnIdle(self, event):
        if self.reInitBuffer:
            self.InitBuffer()
            self.DrawShapes()
            self.Refresh(False)

    def OnPaint(self, event):
        if self.buffer:
            dc = wx.BufferedPaintDC(self, self.buffer)
            if self.originalSize is None:
                self.originalSize = dc.GetSize()
                print('originalSize', self.originalSize)


if __name__ == '__main__':
    class MtlCanvasFrame(wx.Frame):
        def __init__(self, parent):
            wx.Frame.__init__(self, parent, -1, "Mtl Canvas Frame",
                    size=(800,600))
            self.canvas = MtlCanvas(self, -1)

    app = wx.App(redirect=False)
    frame = MtlCanvasFrame(None)
    frame.canvas.AddShape('rect', 0, 0, 750, 550)
    frame.canvas.AddShape('circle', 50, 50, 50)
    frame.canvas.AddShape('circle', 375, 275, 75)
    frame.canvas.AddShape('circle', 650, 450, 100)
    frame.Show(True)
    app.MainLoop()
