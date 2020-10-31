#!/usr/bin/env python3

# Convert MMTL output to PUL.DAT format for CR Paul Fortran programs

import wx
import sys
import re


class Frame(wx.Frame):
    """Frame class."""
    def __init__(self, parent=None, id=-1,
                 pos=wx.DefaultPosition,
                 title='MMTL to FOR'):

        wx.Frame.__init__(self, parent, id, title, pos)
        panel = wx.Panel(self, -1)
        panel.SetBackgroundColour("White")
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.SetMinSize((800, 600))

        self.createMenuBar()
        btnSizer = self.createButtonBar(panel, mainSizer)
        textSizer = self.createTextFields(panel, mainSizer)

        mainSizer.Add(btnSizer, 0, wx.EXPAND | wx.BOTTOM)
        mainSizer.Add(textSizer, 1, wx.EXPAND | wx.BOTTOM)

        panel.SetSizer(mainSizer)
        mainSizer.Fit(self)

    def buttonData(self):
        return (("First", self.OnFirst),
                ("<< PREV", self.OnPrev),
                ("NEXT >>", self.OnNext),
                ("Last", self.OnLast))

    def menuData(self):
        return (("&File",
                 ("&Open", "Open in status bar", self.OnOpen),
                 ("&Quit", "Quit", self.OnCloseWindow)),
                ("&Edit",
                 ("&Copy", "Copy", self.OnCopy),
                 ("C&ut", "Cut", self.OnCut),
                 ("&Paste", "Paste", self.OnPaste),
                 ("", "", ""),
                 ("&Options...", "DisplayOptions", self.OnOptions)))

    def textFieldData(self):
        return (('Inductance (H/m)', 'inductance'),
                ('Capacitance (F/m)', 'capacitance'))

    def createButtonBar(self, panel, mainSizer):
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        for eachLabel, eachHandler in self.buttonData():
            button = self.buildOneButton(panel, eachLabel, eachHandler)
            btnSizer.Add(button, flag=wx.ALL | wx.EXPAND, border=20)

        return btnSizer

    def buildOneButton(self, parent, label, handler):
        button = wx.Button(parent, -1, label)
        self.Bind(wx.EVT_BUTTON, handler, button)

        return button

    def createMenuBar(self):
        menuBar = wx.MenuBar()
        for eachMenuData in self.menuData():
            menuLabel = eachMenuData[0]
            menuItems = eachMenuData[1:]
            menuBar.Append(self.createMenu(menuItems), menuLabel)
        self.SetMenuBar(menuBar)

    def createMenu(self, menuData):
        menu = wx.Menu()
        for eachLabel, eachStatus, eachHandler in menuData:
            if not eachLabel:
                menu.AppendSeparator()
                continue
            menuItem = menu.Append(-1, eachLabel, eachStatus)
            self.Bind(wx.EVT_MENU, eachHandler, menuItem)
        return menu

    def createTextFields(self, panel, mainSizer):
        textSizer = wx.BoxSizer(wx.VERTICAL)

        for data in self.textFieldData():
            label, field = self.createCaptionedText(panel, data,
                                                    style=wx.TE_MULTILINE)
            fieldSizer = wx.BoxSizer(wx.HORIZONTAL)
            fieldSizer.Add(label, 0, flag=wx.ALL | wx.EXPAND, border=10)
            fieldSizer.Add(field, 1, flag=wx.ALL | wx.EXPAND, border=10)
            textSizer.Add(fieldSizer, 1, flag=wx.EXPAND)

        return textSizer

    def createCaptionedText(self, panel, data, style=0):
        static = wx.StaticText(panel, wx.ID_ANY, data[0])
        static.SetBackgroundColour("White")
        text = wx.TextCtrl(panel, wx.ID_ANY, "", name=data[1], style=style)

        return [static, text]

    def OnPrev(self, event): pass
    def OnNext(self, event): pass
    def OnLast(self, event): pass

    def OnFirst(self, event):
        indText = wx.FindWindowByName('inductance').GetValue()
        capText = wx.FindWindowByName('capacitance').GetValue()
        getMatrices(indText, capText)

    def OnOpen(self, event): pass
    def OnCopy(self, event): pass
    def OnCut(self, event): pass
    def OnPaste(self, event): pass
    def OnOptions(self, event): pass

    def OnCloseWindow(self, event):
        self.Destroy()


class App(wx.App):
    """Application class."""
    def OnInit(self):
        self.frame = Frame(parent=None, id=-1, title='MMTL to FOR')
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

    def OnExit(self):
        return True


def parseStrMatrix(text):
    lines = text.split('\n')
    matrix = []
    for line in lines:
        cols = re.findall(r"[^ \t]+", line)
        matrix.append(cols)

    return matrix


def convertStrToNum(strMatrix):
    matrix = []
    for row in range(len(strMatrix)):
        matrix.append([])
        for col in range(len(strMatrix[row])):
            val = float(strMatrix[row][col])
            matrix[row].append(val)

    return matrix


def isDiagonal(matrix):
    N = len(matrix)
    for i in range(N):
        if len(matrix[i]) is not (i+1):
            return False

    return True


def getMatrix(text):
    strmat = parseStrMatrix(text)
    nummat = convertStrToNum(strmat)
    if isDiagonal(nummat):
        return nummat
    else:
        return None


def getMatrices(indText, capText):
    ind_mat = getMatrix(indText)
    cap_mat = getMatrix(capText)

    print('Inductance (H/m)')
    print(ind_mat)
    print('Capacitance (F/m)')
    print(cap_mat)

    printPulFile(ind_mat, cap_mat)


def printPulFile(ind_mat, cap_mat):
    printPulMatrix(ind_mat, 'L')
    printPulMatrix(cap_mat, 'C')


def printPulMatrix(mat, label):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            text = ' {0}    {1}    {2:.5e}    ={3}({4},{5})'.format(
                (i+1), (j+1), mat[i][j], label, (i+1), (j+1))
            print(text)


if __name__ == '__main__':
    app = App(redirect=False)
    app.MainLoop()
