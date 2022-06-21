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
Provides helper function for selecting part of a system, e.g. all proteins, or
protein backbones.
"""

import numpy as np
from .molecule import attributes_match


# TODO: Make that list part of the force fields
PROTEIN_RESIDUES = set(('ALA', 'ARG', 'ASP', 'ASN', 'CYS',
                        'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                        'SER', 'THR', 'TRP', 'TYR', 'VAL',
                        'HSE', 'HIE', 'HSD', 'HID', 'HSP',
                        'HIP', 'ASPP', 'GLUP', 'LSN', 'ASH',
                        'GLH', 'LYN'))


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


def select_all(_):
    """
    Returns True for all particles.
    """
    return True


# TODO: Have the backbone definition be force field specific.
def select_backbone(node):
    """
    Returns True if `node` is in a protein backbone.
    """
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


def proto_select_attribute_in(node, attribute, values):
    """
    Return True if the given attribute of the node is in a list of values.

    To be used as a selector, the function must be wrapped in a way that it can
    be called without the need to explicitly specify the 'attribute' and
    'values' arguments. This can be done using :func:`functools.partial`:

    >>> # select an atom if its name is in a given list
    >>> to_keep = ['BB', 'SC1']
    >>> select_name_in = functools.partial(
    ...     proto_select_attribute_in,
    ...     attribute='atomname',
    ...     values=to_keep
    ... )
    >>> select_name_in(node)

    Parameters
    ----------
    node: dict
        The atom/node to consider.
    attribute: str
        The key to look at in the node.
    values: list
        The values the node attribute can take for the node to be selected.

    Returns
    -------
    bool
    """
    return node.get(attribute) in values


def proto_multi_templates(node, templates, ignore_keys=()):
    """
    Return `True` is the node matched one of the templates.

    Parameters
    ----------
    node: dict
        The atom/node to consider.
    templates: collections.abc.Iterable[dict]
        A list of node templates to compare to the node.
    ignore_keys: collections.abc.Collection
        List of keys to ignore from the templates.

    Returns
    -------
    bool

    See Also
    --------
    vermouth.molecule.attributes_match
    """
    return any(
        attributes_match(node, template, ignore_keys)
        for template in templates
    )


def filter_minimal(molecule, selector):
    """
    Yield the atom keys that match the selector.

    The selector must be a function that accepts an atom as a argument. The
    atom is passed as a node attribute dictionary. The selector must return
    ``True`` for atoms to keep in the selection.

    The function can be used to build a subgraph that only contains the
    selection:

    .. code-block:: python

        selection = molecule.subgraph(
             filter_minimal(molecule, selector_function)
        )

    Parameters
    ----------
    molecule: Molecule
    selector: collections.abc.Callable

    Yields
    ------
    collections.abc.Hashable:
        Keys of the atoms that match the selection.
    """
    for name, atom in molecule.nodes.items():
        if selector(atom):
            yield name
