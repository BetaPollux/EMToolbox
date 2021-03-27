#!/bin/usr/python3

import numpy as np


class MtlNetwork():
    def __init__(self, tline, source_z, load_z):
        self.tline = tline
        self.source_z = source_z
        self.load_z = load_z

    def reflection(self, f, position=None):
        '''Calculate the reflection coefficient, where 
        position = none returns the coefficient at the load'''
        zc = self.tline.char_impedance()
        refl = (self.load_z - zc) / (self.load_z + zc)
        if position is not None:
            refl *= np.exp(2 * self.tline.prop_const(f) *
                           (position - self.tline.length))
        return refl
    
    def input_impedance(self, f, position=0):
        '''Solve for input impedance at the given position'''
        refl = self.reflection(f, position)
        return self.tline.char_impedance() * (1 + refl) / (1 - refl)
    
    def solve(self, f, source_v):
        '''Solve for forward and backward travelling voltages [Vfwd, Vbwd]'''
        zc = self.tline.char_impedance()
        zs = self.source_z
        zl = self.load_z
        jbl = self.tline.prop_const(f) * self.tline.length
        A = np.array([[zc + zs,
                       zc - zs],
                      [(zc - zl) * np.exp(-jbl),
                       (zc + zl) * np.exp(jbl)]])
        b = np.array([zc * source_v, 0])
        return np.linalg.solve(A, b)

    def get_voltage(self, f, solution, position):
        '''Get voltage along the line, where solution comes from solve()'''
        jbz = self.tline.prop_const(f) * position
        A = np.array([np.exp(-jbz), np.exp(jbz)])
        return A @ solution
    
    def get_current(self, f, solution, position):
        '''Get current along the line, where solution comes from solve()'''
        jbz = self.tline.prop_const(f) * position
        zc = self.tline.char_impedance()
        A = np.array([np.exp(-jbz) / zc, -np.exp(jbz) / zc])
        return A @ solution
    
    def vswr(self, f):
        refl = abs(self.reflection(f))
        if refl < 1:
            return (1 + refl) / (1 - refl)
        else:
            return np.inf
