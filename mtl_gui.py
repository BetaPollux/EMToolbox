#!/usr/bin/python3

import wx


class MtlFrame(wx.Frame):
    '''Top level frame for the multi-conductor transmission line solvers'''
    def __init__(self, parent=None, id=-1,
                 title='Multi-conductor Transmission Lines',
                 pos=wx.DefaultPosition,
                 size=(640, 480)):
        super().__init__(parent, id, title, pos, size)
        self.panel = wx.Panel(self)


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
    app = MtlApp()
    app.MainLoop()
