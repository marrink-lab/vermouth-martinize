# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:34:12 2017

@author: Peter Kroon
"""
from ..molecule import Molecule
from ..utils import first_alpha

from itertools import chain

import numpy as np


def read_gro(file_name, exclude=('SOL',), ignh=False):
    molecule = Molecule()
    idx = 0
    field_types = [int, str, str, int, float, float, float]
    field_names = ['resid', 'resname', 'atomname', 'atomid', 'x', 'y', 'z']
    field_widths = [5, 5, 5, 5]

    with open(file_name) as gro:
        next(gro)  # skip title
        num_atoms = int(next(gro))

        # We need the first line to figure out the exact format. In particular,
        # the precision and whether it has velocities.
        first_line = next(gro)
        has_vel = first_line.count('.') == 6
        first_dot = first_line.find('.', 25)
        second_dot = first_line.find('.', first_dot+1)
        precision = second_dot - first_dot

        field_widths.extend([precision]*3)
        if has_vel:
            field_widths.extend([precision]*3)
            field_types.extend([float]*3)
            field_names.extend(['vx', 'vy', 'vz'])

        start = 0
        slices = []
        for width in field_widths:
            if width > 0:
                slices.append(slice(start, start + width))
            start = start + abs(width)

        # Start parsing the file in earnest. And let's not forget the first
        # line.
        for line_idx, line in enumerate(chain([first_line], gro)):
            properties = {}
            # This (apart maybe from adhering to the number of lines specified
            # by the file) is the fastest method of checking whether we are at
            # the last line (box) of the file. Other things tested: regexp
            # matching, looking ahead, and testing whether the line looks like
            # a box-line. I think the reason this is faster is because the try
            # block will almost never raise an exception.
            try:
                for name, type_, slice_ in zip(field_names, field_types, slices):
                    properties[name] = type_(line[slice_].strip())
            except ValueError:
                if line_idx != num_atoms:
                    raise
                break

            properties['element'] = first_alpha(properties['atomname'])
            properties['chain'] = ''
            if properties['resname'] in exclude or (ignh and properties['element'] == 'H'):
                continue

            pos = (properties.pop('x'), properties.pop('y'), properties.pop('z'))
            properties['position'] = np.array(pos, dtype=float)
            properties['position'] *= 10  # Convert nm to A

            if has_vel:
                vel = (properties.pop('vx'), properties.pop('vy'), properties.pop('vz'))
                properties['velocity'] = np.array(vel, dtype=float)
                properties['velocity'] *= 10  # Convert nm to A

            molecule.add_node(idx, **properties)
            idx += 1
    return molecule
