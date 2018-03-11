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


# TODO: Make that list part of the force fields
PROTEIN_RESIDUES = set(('ALA', 'ARG', 'ASP', 'ASN', 'CYS',
                        'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                        'SER', 'THR', 'TRP', 'TYR', 'VAL'))


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


# TODO: Have the backbone definition be force field specific.
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
    Yield the atom keys that match the selector.

    The selector must be a function that accepts an atom as a argument. The
    atom is passed as a node attribute dictionary. The selector must return
    ``True`` for atoms to keep in the selection.

    The function can be used to build a subgraph that only contains the
    selection:

    .. code:: python
        selection = molecule.subgraph(
             filter_minimal(molecule, selector_function)
        )

    Parameters
    ----------
    molecule: Molecule
    selector: callback

    Yields
    ------
    keys:
        Keys of the atoms that match the selection.
    """
    filtered = Molecule()
    for name, atom in molecule.nodes.items():
        if selector(atom):
            yield name
