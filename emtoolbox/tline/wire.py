#!/usr/bin/python3

import numpy as np


class Wire():
    """Wire with a given location and radius."""

    def __init__(self, x: float, y: float, radius: float,
                 ins_thickness: float = 0.0, ins_er: float = 1.0):
        if radius <= 0:
            raise ValueError('Radius must be greater than 0')
        self.x = x
        self.y = y
        self.radius = radius
        self.ins_thickness = ins_thickness
        self.ins_er = ins_er

    @property
    def ins_radius(self):
        """Return the overall insulation radius."""
        return self.radius + self.ins_thickness

    def distance_to(self, wire):
        """Center-to-center distance between wires."""
        return np.sqrt((self.x - wire.x)**2 + (self.y - wire.y)**2)

    def gap_to(self, wire):
        """Gap between external wire surfaces."""
        return self.distance_to(wire) - self.radius - wire.radius

    def angle_to(self, wire):
        """Angle in radians to other wire, with CCW being positive."""
        return np.arctan2(wire.y, wire.x) - np.arctan2(self.y, self.x)

    def offset(self, x: float = 0, y: float = 0):
        """Offset from position."""
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)


class Plane():
    """Infinite plane located at (x, 0)."""

    def __init__(self, width: float = 0.1, thickness: float = 0.1e-3):
        self.width = width
        self.thickness = thickness

    def height_of(self, wire):
        """Return height of wire above plane."""
        return wire.y


class Shield():
    """Circular shield with a fixed radius, located at (0, 0)."""

    def __init__(self, radius: float, thickness: float = 0.5e-3):
        if radius <= 0:
            raise ValueError('Radius must be greater than 0')
        self.radius = radius
        self.thickness = thickness

    def contains(self, wire):
        """Return whether the wire is fully inside the shield."""
        return self.radius >= wire.offset() + wire.radius
