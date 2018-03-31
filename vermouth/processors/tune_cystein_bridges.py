#!/usr/bin/env python3
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

import functools
from ..molecule import attributes_match
from .. import selectors

UNIVERSAL_BRIDGE_TEMPLATE = {'resname': 'CYS', 'atomname': 'SG1'}


def prune_edges_between_selections(molecule, selection_a, selection_b):
    """
    Remove edges which have their ends part of given selections.

    An edge is removed if has one end that is part of 'selection_a', and the
    other end part of 'selection_b'.

    Parameters
    ----------
    molecule: nx.Graph
        Molecule to prune in-place.
    selection_a, selection_b: list
        Lists of node keys from the molecule.

    See Also
    --------
    prune_edges_with_selectors
    """
    selection_a = set(selection_a)
    selection_b = set(selection_b)
    # to_prune is the list of edges to remove. It cannot be a generator or
    # removing the edges will be modifying the datastructure on which we
    # iterate.
    to_prune = [
        edge for edge in molecule.edges
        if ((edge[0] in selection_a and edge[1] in selection_b)
                or (edge[1] in selection_a and edge[0] in selection_b))
    ]
    molecule.remove_edges_from(to_prune)


def prune_edges_with_selectors(molecule, selector_a, selector_b=None):
    """
    Remove edges with the ends between selections defined by selectors.

    An edge is removed if one of its end is part of the selection defined by
    'selector_a', and its other end is part of the selection defined by
    'selector_b'. A selector is a function that accept a node dictionary as
    argument and returns ``True`` if the node is part of the selection.

    The 'selection_b' argment is optional. If it is ``None``, then 'selector_a'
    is used for the selection at both ends.

    Parameters
    ----------
    molecule: nx.Graph
        Molecule to prune in-place.
    selector_a: function
        A selector for one end of the edges.
    selector_b: function (optional)
        A selector for the second end of the edges. If set to ``None``, then
        'selector_a' is used for both ends.

    See Also
    --------
    prune_edges_between_selections
    """
    selection_a = selectors.filter_minimal(molecule, selector_a)
    if selector_b is None:
        selector_b = selector_a
    selection_b = selectors.filter_minimal(molecule, selector_b)
    prune_edges_between_selections(molecule, selection_a, selection_b)


def remove_cystein_bridge_edges(molecule, template=UNIVERSAL_BRIDGE_TEMPLATE):
    """
    Remove all the edges that correspond to cystein bridges from a molecule.

    Cystein bridge edges link an atom from a cystein side chain to the same
    atom on an other cystein. Selecting the correct atom is done with a
    template node dictionary, in the same way as node matching in links. The
    default template selects the 'SG1' bead of the residue 'CYS':
    ``{'resname': 'CYS', 'atomname': 'SG1'}``.

    A template is a dictionary that defines the key:value pairs that must be
    matched in the atoms. Values can be instances of
    :class:`molecule.LinkPredicate`.

    Parameters
    ----------
    molecule: nx.Graph
        Molecule to modify in-place.
    template: dict (optional)
        A template that selected atom must match.
    """
    selector = functools.partial(attributes_match, template_attributes=template)
    prune_edges_with_selectors(molecule, selector)


class RemoveCysteinBridgeEdges(Processor):
    def __init__(self, template=UNIVERSAL_BRIDGE_TEMPLATE):
        self.template = UNIVERSAL_BRIDGE_TEMPLATE

    def run_molecule(self, molecule):
        remove_cystein_bridge_edges(molecule, self.template)
        return molecule
