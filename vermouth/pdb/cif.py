# -*- coding: utf-8 -*-
# Copyright 2025 University of Groningen
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
Provides functionality to read CIF files.
"""

try:
    from CifFile import ReadCif
except ImportError:
    HAVE_READCIF = False
else:
    HAVE_READCIF= True

import numpy as np

from ..molecule import Molecule

def _cell(cf, modelname):
    """
    get the cell dimensions from a cif file as a list
    """
    a = cf[modelname]['_cell.length_a']
    b = cf[modelname]['_cell.length_b']
    c = cf[modelname]['_cell.length_c']
    alpha = cf[modelname]['_cell.angle_alpha']
    beta = cf[modelname]['_cell.angle_beta']
    gamma = cf[modelname]['_cell.angle_gamma']

    return np.array([a, b, c, alpha, beta, gamma], dtype=np.float32)

def read_cif_file(file_name, exclude=('SOL', 'HOH'), ignh=False):
    """
    Parse a CIF file to create a molecule using the PyCIFRW library

    Parameters
    ----------
    filename: str
        The file to read.
    exclude: collections.abc.Container[str]
        Atoms that have one of these residue names will not be included.
    ignh: bool
        Whether hydrogen atoms should be ignored.
    model: int
        If the PDB file contains multiple models, which one to select.

    Returns
    -------
    list[vermouth.molecule.Molecule]
        The parsed molecules. Will only contain edges if the PDB file has
        CONECT records. Either way, the molecules might be disconnected. Entries
        separated by TER, ENDMDL, and END records will result in separate
        molecules.
    """

    # list of categories from the read cif file to read into a molecule
    cif_categories = [
    '_atom_site.id', # atomid
    '_atom_site.auth_atom_id', # atomname
    '_atom_site.auth_comp_id', # resname
    '_atom_site.auth_asym_id', # chain
    '_atom_site.auth_seq_id', # resid
    #'_atom_site.pdbx_pdb_ins_code', # insertion_code
    '_atom_site.cartn_x', # x
    '_atom_site.cartn_y', # y
    '_atom_site.cartn_z', # z
    #'_atom_site.occupancy', # occupancy
    #'_atom_site.b_iso_or_equiv', # temp_factor
    '_atom_site.type_symbol', # element
    '_atom_site.pdbx_formal_charge', # charge
    ]
    cif_category_names = ['atomid', 'atomname', 'resname', 'chain', 'resid',
                          'x', 'y', 'z',
                          'element', 'charge']
    cif_category_types = [int, str, str, str, int,
                          float, float, float,
                          str, str]

    cf = ReadCif(str(file_name))

    molecules = []

    for model in cf.keys():

        molecule = Molecule()
        # annoyingly _refine_hist.number_atoms_total is not a mandatory entry
        number_atoms = len(cf[model]['_atom_site.cartn_x'])

        for idx in range(number_atoms):
            properties = {}

            for attr, attr_type, cif_key in zip(cif_category_names, cif_category_types, cif_categories):
                properties[attr] = attr_type(cf[model][cif_key][idx])

            if properties['resname'] in exclude or (ignh and properties['element'] == 'H'):
                continue

            properties['position'] = np.array([properties['x'],
                                               properties['y'],
                                               properties['z'],
                                               ], dtype=np.float32) / 10

            molecule.add_node(idx, **properties)

        molecule.box = _cell(cf, model)
        molecules.append(molecule)

    return molecules

class CIFReader():
    def __init__(self, file, exclude, ignh):
        self.input_file = file
        self.exclude = exclude
        self.ignh = ignh

    def reader(self):
        if HAVE_READCIF:
            molecules = read_cif_file(self.input_file, self.exclude, self.ignh)
            return molecules
        else:
            raise ImportError("The PyCifRW library must be installed to read .cif files with Vermouth.")

