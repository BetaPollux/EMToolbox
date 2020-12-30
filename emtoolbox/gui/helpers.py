#!/usr/bin/python3

'''Helper functions for GUIs'''

import sys
import wx


def parse_input_fields(parent: wx.Window, fields: list) -> dict:
    '''Find text input fields by name and collect their values.
    Fields must be derived from wx.TextEntry
    Arguments:
        parent: The parent window
        fields: (name, *args)
    Return:
        result: { name: value (str) }'''
    result = {}
    for name, *_ in fields:
        field = parent.FindWindowByName(name)
        if field and isinstance(field, wx.TextEntry):
            result[name] = field.GetValue()
        else:
            print(f'Could not parse input field: {name}',
                  file=sys.stderr)
    return result


def populate_output_fields(parent: wx.Window, fields: dict) -> None:
    '''Find text fields by name and set their text to str(value).
    Fields must be derived from wx.TextEntry
    Arguments:
        parent: The parent window
        fields: { name: value }'''
    for name, value in fields.items():
        field = parent.FindWindowByName(name)
        if field and isinstance(field, wx.TextEntry):
            field.SetValue(str(value))
        else:
            print(f'Could not populate output field: {name}, {value}',
                  file=sys.stderr)


def create_text_field_set(parent: wx.Window, fields: list,
                          text_style: int = 0) -> wx.Sizer:
    '''Create labeled text fields and insert into a sizer.
    Arguments:
        parent:     The parent window
        fields:     [(name, label, default), ...]
                name:       name for TextCtrl
                label:      text for StaticText
                default:    default text for TextCtrl
        text_style: style for TextCtrl
    Return:
        result: wx.Sizer containing the new windows'''
    text_field_sizer = wx.FlexGridSizer(cols=2, vgap=5, hgap=5)
    text_field_sizer.AddGrowableCol(1, 1)
    for name, label, default in fields:
        static, text = create_text_field(parent,
                                         name, label, default,
                                         text_style)
        text_field_sizer.Add(static, 0, wx.ALIGN_CENTER_VERTICAL)
        text_field_sizer.Add(text, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
    return text_field_sizer


def create_text_field(parent: wx.Window, name: str, label: str,
                      default: str = '', text_style: int = 0) -> tuple:
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
                            combo_style: int = 0) -> wx.Sizer:
    '''Create labeled choice fields and insert into a sizer.
    Arguments:
        parent:     The parent window
        fields:     [(name, label, choices), ...]
                name:       name for ComboBox
                label:      text for StaticText
                choices:    choices for ComboBox (list)
        combo_style: style for ComboBox
    Return:
        result: wx.Sizer containing the new windows'''
    choices_field_sizer = wx.FlexGridSizer(cols=2, vgap=5, hgap=5)
    choices_field_sizer.AddGrowableCol(1, 1)
    for name, label, choices in fields:
        static, combo = create_choice_field(parent,
                                            name, label, choices,
                                            combo_style)
        choices_field_sizer.Add(static, 0, wx.ALIGN_CENTER_VERTICAL)
        choices_field_sizer.Add(combo, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
    return choices_field_sizer


def create_choice_field(parent: wx.Window, name: str, label: str,
                        choices: list, combo_style: int = 0) -> tuple:
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
