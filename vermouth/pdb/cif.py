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
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


def casting(value, typeto):
    try:
        return typeto(value)
    # except a ValueError. The two known ones are resid = '.' or charge = '?' which must be a str.
    # may cause an error if more are added
    except ValueError:
        return None

def _cell(cf, modelname):
    """
    get the cell dimensions from a cif file as a list
    """
    try:
        a = cf[modelname]['_cell.length_a']
        b = cf[modelname]['_cell.length_b']
        c = cf[modelname]['_cell.length_c']
        alpha = cf[modelname]['_cell.angle_alpha']
        beta = cf[modelname]['_cell.angle_beta']
        gamma = cf[modelname]['_cell.angle_gamma']
        dims = np.array([a, b, c, alpha, beta, gamma], dtype=float)
    except KeyError:
        LOGGER.info("_cell information missing from .cif file. Will write default dimensions")
        dims = np.array([1, 1, 1, 90, 90, 90])

    return dims


def read_cif_file(file_name, exclude=('SOL', 'HOH'), ignh=False, modelidx=1):
    """
    Parse a CIF file to create a molecule using the PyCIFRW library

    Parameters
    ----------
    file_name: str
        The file to read.
    exclude: collections.abc.Container[str]
        Atoms that have one of these residue names will not be included.
    ignh: bool
        Whether hydrogen atoms should be ignored.
    modelidx: int
        If the cif file contains multiple models, which one to select.

    Returns
    -------
    list[vermouth.molecule.Molecule]
        The parsed molecules. Will not contain edges
    """

    # list of categories from the read cif file to read into a molecule
    cif_categories_all = [
        '_atom_site.id',  # atomid
        '_atom_site.label_atom_id',  # atomname
        '_atom_site.label_comp_id',  # resname
        '_atom_site.label_asym_id',  # chain
        '_atom_site.label_seq_id',  # resid
        '_atom_site.pdbx_pdb_ins_code',  # insertion_code
        '_atom_site.cartn_x',  # x
        '_atom_site.cartn_y',  # y
        '_atom_site.cartn_z',  # z
        '_atom_site.occupancy',  # occupancy
        '_atom_site.b_iso_or_equiv',  # temp_factor
        '_atom_site.type_symbol',  # element
        '_atom_site.pdbx_formal_charge',  # charge (nb not added by AF3)
        '_atom_site.pdbx_PDB_model_num'  # model
    ]
    cif_category_names = ['atomid', 'atomname', 'resname', 'chain', 'resid', 'insertion_code',
                          'x', 'y', 'z',
                          'occupancy', 'temp_factor',
                          'element', 'charge', 'model']
    cif_category_types = [int, str, str, str, int, str,
                          float, float, float,
                          float, float,
                          str, float, int]
    essential_properties = ['resname', 'resid']

    cf = ReadCif(str(file_name))
    # PyCifRW seems to store everything from the file it reads in a top level key from _entry.id
    # which ~ corresponds to the pdb code. From a single file we should only have one key.
    if len(cf.keys()) > 1:
        LOGGER.warning("This cif file contains multiple entries. Will parse the first one.")
    fname = list(cf.keys())[0]

    # first filter the data by which categories are present
    # make list of the category names which are present
    names = []
    # make a list of the lists of the data
    all_data = []
    for category, name, cattype in zip(cif_categories_all, cif_category_names, cif_category_types):
        if category in cf[fname]:
            # cast the data to its correct type
            data = [casting(i, cattype) for i in cf[fname][category]]
            all_data.append(data)
            names.append(name)
        # if we're missing one of the essential categories, raise a warning
        elif name in essential_properties:
            LOGGER.warning(f"{name} data is missing from the input file, and is a required field.")

    # if we don't have atomnames but do have element information, copy it.
    if ('atomname' not in names) and ('element' in names):
        LOGGER.warning("atomname data missing from input file. Will attempt to continue using element data.")
        names.append('atomname')
        all_data.append([casting(i, str) for i in cf[fname]['_atom_site.type_symbol']])

    # for each atom, make a dictionary with its associated name
    properties_dict_list = [dict(zip(names, v)) for v in zip(*all_data)]

    molecules = []

    molecule = Molecule()

    # find the indices in the data which are actually from the model that we're after.
    model_atom_indices = [i for i, x in enumerate([int(j) == modelidx for
                                                   j in cf[fname]['_atom_site.pdbx_PDB_model_num']]) if x]

    idx = 0  # add nodes by separate index in case we skip some atoms
    for _ in model_atom_indices:
        properties = properties_dict_list[_]

        if properties.get('resname', None) in exclude or (ignh and properties['element'] == 'H'):
            continue

        properties['position'] = np.array([properties['x'],
                                           properties['y'],
                                           properties['z'],
                                           ], dtype=float) / 10

        molecule.add_node(idx, **properties)
        idx += 1

    molecule.box = _cell(cf, fname)
    molecules.append(molecule)

    return molecules

class CIFReader():
    def __init__(self, file, exclude, ignh, modelidx=1):
        self.input_file = file
        self.exclude = exclude
        self.ignh = ignh
        self.modelidx = modelidx

    def reader(self):
        if HAVE_READCIF:
            molecules = read_cif_file(self.input_file, self.exclude, self.ignh, self.modelidx)
            return molecules
        else:
            raise ImportError("The PyCifRW library must be installed to read .cif files with Vermouth.")

