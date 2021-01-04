#!/usr/bin/python3

'''Helper functions for GUIs'''

import sys
import wx

def pack_input_params(name: str, label: str, default: str = '') -> dict:
    '''Pack input field parameters into a dictionary,
    as required by the create field set functions'''
    return {'name': name,
            'label': label,
            'default': default}

def pack_choice_item(name: str, label: str, choices: list) -> dict:
    '''Pack choice field parameters into a dictionary,
    as required by the create field set functions'''
    return {'name': name,
            'label': label,
            'choices': choices}

def parse_input_fields(parent: wx.Window, fields: list, convert: type = None) -> dict:
    '''Find text input fields by name and collect their values.
    Fields must be derived from wx.TextEntry
    Arguments:
        parent: The parent window
        fields: [{ name: window name}]
        type:   convert values to the provided type
    Return:
        result: { name: value (str) }'''
    result = {}
    for field in fields:
        window = parent.FindWindowByName(field['name'])
        if window and isinstance(window, wx.TextEntry):
            if convert:
                result[field['name']] = convert(window.GetValue())
            else:
                result[field['name']] = window.GetValue()
        else:
            raise Exception(f'Could not parse input field: {field["name"]}')
    return result


def populate_output_fields(parent: wx.Window, fields: dict) -> None:
    '''Find text fields by name and set their text to str(value).
    Fields must be derived from wx.TextEntry
    Arguments:
        parent: The parent window
        fields: { name: value }'''
    for name, value in fields.items():
        window = parent.FindWindowByName(name)
        if window and isinstance(window, wx.TextEntry):
            window.SetValue(str(value))
        else:
            raise Exception(f'Could not populate output field: {name}, {value}')


def create_text_field_set(parent: wx.Window, fields: list,
                          text_style: int = 0) -> list:
    '''Create labeled text fields and insert into a sizer.
    Arguments:
        parent:     The parent window
        fields:     [{name, label, default}, ...]
                name:       name for TextCtrl
                label:      text for StaticText
                default:    default text for TextCtrl
        text_style: style for TextCtrl
    Return:
        result: list of new (StaticText, TextCtrl)'''
    return [create_text_field(parent, **field, text_style=text_style)
            for field in fields]


def create_text_field(parent: wx.Window, name: str, label: str,
                      default: str = '', text_style: int = 0, **_) -> tuple:
    '''Creates and returns a (StaticText, TextCtrl).
    Arguments:
        name:       name for TextCtrl
        label:      text for StaticText
        default:    default text for TextCtrl
        text_style: style for TextCtrl'''
    static = wx.StaticText(parent, wx.ID_ANY, label=label)
    text = wx.TextCtrl(parent, wx.ID_ANY,
                       value=str(default), name=name,
                       style=text_style)
    return (static, text)


def create_choice_field_set(parent: wx.Window, fields: list,
                            combo_style: int = 0) -> list:
    '''Create labeled choice fields and insert into a sizer.
    Arguments:
        parent:     The parent window
        fields:     [{name:, label, choices}, ...]
                name:       name for ComboBox
                label:      text for StaticText
                choices:    choices for ComboBox (list)
        combo_style: style for ComboBox
    Return:
        result: list of (StaticText, ComboBox)'''
    return [create_choice_field(parent, **field, combo_style=combo_style)
            for field in fields]


def create_choice_field(parent: wx.Window, name: str, label: str,
                        choices: list, combo_style: int = 0, **_) -> tuple:
    '''Creates and returns a (StaticText, ComboBox).
    Arguments:
        name:           name for ComboBox
        label:          text for StaticText
        choices:        choices for ComboBox
        combo_style:    style for ComboBox'''
    static = wx.StaticText(parent, wx.ID_ANY, label=label)
    combo = wx.ComboBox(parent, id=wx.ID_ANY, choices=choices,
                        style=combo_style, name=name)
    combo.SetSelection(0)
    return (static, combo)


def layout_buttons(buttons: list) -> wx.Sizer:
    '''Create a sizer containing all buttons
    Arguments:
        buttons: [btn0, btn1, ...]
    Return:
        Sizer containing the buttons'''
    btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
    for btn in buttons:
        btn_sizer.Add(btn, 0, wx.ALL, 5)
    return btn_sizer

def layout_fields(fields: list):
    '''Create a sizer containing all fields
    Arguments:
        fields: [(label, field), ...]
    Return:
        Sizer containing the labeled fields'''
    field_sizer = wx.FlexGridSizer(cols=2)
    field_sizer.AddGrowableCol(1)
    for label, field in fields:
        field_sizer.Add(label, 0, wx.LEFT | wx.BOTTOM, 5)
        field_sizer.Add(field, 0, wx.EXPAND | wx.LEFT | wx.BOTTOM | wx.RIGHT, 5)
    return field_sizer
