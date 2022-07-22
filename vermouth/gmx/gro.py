# -*- coding: utf-8 -*-
# Copyright 2018 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Provides functionality to read and write GRO96 files.
"""

from itertools import chain

import numpy as np

from ..file_writer import deferred_open
from ..molecule import Molecule
from ..truncating_formatter import TruncFormatter
from ..utils import first_alpha


def read_gro(file_name, exclude=('SOL',), ignh=False):
    """
    Parse a gro file to create a molecule.

    Parameters
    ----------
    filename: str
        The file to read.
    exclude: collections.abc.Container[str]
        Atoms that have one of these residue names will not be included.
    ignh: bool
        Whether hydrogen atoms should be ignored.

    Returns
    -------
    vermouth.molecule.Molecule
        The parsed molecules. Will not contain edges.
    """
    molecule = Molecule()
    idx = 0
    field_types = [int, str, str, int, float, float, float]
    field_names = ['resid', 'resname', 'atomname', 'atomid', 'x', 'y', 'z']
    field_widths = [5, 5, 5, 5]

    with open(str(file_name)) as gro:
        next(gro)  # skip title
        num_atoms = int(next(gro))

        # We need the first line to figure out the exact format. In particular,
        # the precision and whether it has velocities.
        first_line = next(gro)
        has_vel = first_line.count('.') == 6
        first_dot = first_line.find('.', 25)
        second_dot = first_line.find('.', first_dot + 1)
        precision = second_dot - first_dot

        field_widths.extend([precision] * 3)
        if has_vel:
            field_widths.extend([precision] * 3)
            field_types.extend([float] * 3)
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

            if has_vel:
                vel = (properties.pop('vx'), properties.pop('vy'), properties.pop('vz'))
                properties['velocity'] = np.array(vel, dtype=float)

            molecule.add_node(idx, **properties)
            idx += 1
    return molecule


def write_gro(system, file_name, precision=7, title='Martinized!', box=(0, 0, 0), defer_writing=True):
    """
    Write `system` to `file_name`, which will be a GRO96 file.

    Parameters
    ----------
    system: vermouth.system.System
        The system to write.
    file_name: str
        The file to write to.
    precision: int
        The desired precision for coordinates and (optionally) velocities.
    title: str
        Title for the gro file.
    box: tuple[float]
        Box length and optionally angles.
    defer_writing: bool
        Whether to use :meth:`~vermouth.file_writer.DeferredFileWriter.write` for writing data
    """
    formatter = TruncFormatter()
    pos_format_string = '{{:{ntx}.3ft}}'.format(ntx=precision + 1)
    format_string = '{:5dt}{:<5st}{:>5st}{:5dt}' + pos_format_string*3
    # Pick an arbitrary node from the first molecule to see if all molecules
    # have velocities. Somehow I don't think we can write velocities for some
    # molecules but not others...
    has_vel = all('velocity' in next(iter(mol.nodes.values())) for mol in system.molecules)
    if has_vel:
        vel_format_string = '{{:{ntx}.4ft}}'*3
        vel_format_string = vel_format_string.format(ntx=precision+1)

    if defer_writing:
        open = deferred_open
    else:
        from builtins import open
    with open(str(file_name), 'w') as out:
        out.write(title + '\n')  # Title
        out.write(formatter.format('{}\n', system.num_particles))  # number of atoms
        atomid = 1
        for molecule in system.molecules:
            node_order = molecule.nodes
            for node_idx in node_order:
                node = molecule.nodes[node_idx]
                atomname = node['atomname']
                resname = node['resname']
                resid = node['resid']
                x, y, z = node['position']  # pylint: disable=invalid-name

                line = formatter.format(format_string, resid, resname, atomname,
                                        atomid, x, y, z)
                if has_vel:
                    vx, vy, vz = node['velocity']  # pylint: disable=invalid-name
                    line += formatter.format(vel_format_string, vx, vy, vz)
                atomid += 1
                out.write(line + '\n')
        # Box
        out.write(' '.join(str(value) for value in box))
        # to appease VMD which cannot read the file otherwise.
        out.write('\n')
