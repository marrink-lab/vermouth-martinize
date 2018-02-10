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

from .molecule import Molecule


PROTEIN_RESIDUES = ('ALA', 'ARG', 'ASP', 'ASN', 'CYS',
                    'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL')


def is_protein(molecule):
    """
    Return True is all the residues in the molecule are protein residues.

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
    for atom in molecule.nodes.values():
        resname = atom.get('resname')
        if resname not in PROTEIN_RESIDUES:
            return False
    return True


def select_all(node):
    return True


def select_backbone(node):
    return node.get('atomname') == 'BB'


def selector_no_position(atom):
    """
    Return True if the atom does not have a position.

    An atom is considered as not having a position if:
    * the "position" key is not defined;
    * the value of "position" is ``None``.

    Parameters
    ----------
    atom: dict

    Returns
    -------
    bool
    """
    return atom.get('position') is None


def filter_out(molecule, selector):
    """
    Create a minimal molecule which **exclude** a selection.

    The minimal molecule only has nodes. It does dot have any molecule level
    attribute, not does it have edges.

    The selector must be a function that accepts an atom as a argument. The
    atom is passed as a node attribute dictionary. The selector must return
    ``True`` for atoms that are part of the selection to **exclude**.

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
        if not selector(atom):
            filtered.add_node(name, **atom)
    return filtered
