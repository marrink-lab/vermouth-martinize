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

import numpy as np
from .molecule import Molecule


PROTEIN_RESIDUES = ('ALA', 'ARG', 'ASP', 'ASN', 'CYS',
                    'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL')


def is_protein(molecule):
    """
    Return True if all the residues in the molecule are protein residues.

    The function tests if the residue name of all the atoms in the input
    molecule are in ``PROTEIN_RESIDUES``.

    Parameters
    ----------
    molecule: Molecule
        The molecule to test.

    Returns
    -------
    bool
    """
    return all(
        molecule.nodes[n_idx].get('resname') in PROTEIN_RESIDUES
        for n_idx in molecule
    )


def select_all(node):
    return True


def select_backbone(node):
    return node.get('atomname') == 'BB'


def selector_has_position(atom):
    """
    Return True if the atom have a position.

    An atom is considered as not having a position if:
    * the "position" key is not defined;
    * the value of "position" is ``None``;
    * the coordinates are not finite numbers.

    Parameters
    ----------
    atom: dict

    Returns
    -------
    bool
    """
    position = atom.get('position')
    return position is not None and np.all(np.isfinite(position))


def filter_minimal(molecule, selector):
    """
    Create a minimal molecule which only include a selection.

    The minimal molecule only has nodes. It does dot have any molecule level
    attribute, not does it have edges.

    The selector must be a function that accepts an atom as a argument. The
    atom is passed as a node attribute dictionary. The selector must return
    ``True`` for atoms to keep in the selection.

    Parameters
    ----------
    molecule: Molecule
    selector: callback

    Returns
    -------
    Molecule
    """
    filtered = Molecule()
    for name, atom in molecule.nodes.items():
        if selector(atom):
            filtered.add_node(name, **atom)
    return filtered
