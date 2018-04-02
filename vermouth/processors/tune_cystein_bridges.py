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
import numpy as np
import networkx as nx
from .. import KDTree
from ..molecule import attributes_match
from .. import selectors
from .. import geometry
from .processor import Processor

from pprint import pprint

UNIVERSAL_BRIDGE_TEMPLATE = {'resname': 'CYS', 'atomname': 'SG'}


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
    default template selects the 'SG' bead of the residue 'CYS':
    ``{'resname': 'CYS', 'atomname': 'SG'}``.

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


def add_edges_at_distance(molecule, threshold,
                          selection_a, selection_b, attribute='position'):
    """
    Add edges within a molecule when the distance is below a threshold.

    Create edges within a molecule between nodes that have an end part of
    'selection_a', the other end part of 'selection_b', and a distance between
    the ends that is letter than the given threshold.

    All nodes that are part of 'selection_a' or 'selection_b' must have a
    position store under the attribute which key is given with the 'attribute'
    argument. That key is 'position' by default. If at least one node is
    missing a :exc:`KeyError` is raised.

    Parameters
    ----------
    molecule: nx.Graph
        Molecule to modify in-place.
    threshold: float
        The distance threshold under which edges will be created. The distance
        is expressed in nm.
    selection_a, selection_b: list
        Lists of node keys from the molecule.
    attribute: str (optional)
        Name of the key in the node dictionaries under which the coordinates
        are stored.

    Raises
    ------
    KeyError
        At least one node from the selections does not have a position.
    """
    selection_a = set(selection_a)
    selection_b = set(selection_b)
    coordinates_a = np.stack([
        node[attribute]
        for key, node in molecule.nodes.items()
        if key in selection_a
    ])
    coordinates_b = np.stack([
        node[attribute]
        for key, node in molecule.nodes.items()
        if key in selection_b
    ])
    keys_a = np.array([key for key in molecule.nodes.keys() if key in selection_a])
    keys_b = np.array([key for key in molecule.nodes.keys() if key in selection_b])

    distance_matrix = geometry.distance_matrix(coordinates_a, coordinates_b)
    index_a, index_b = np.where(distance_matrix < threshold)
    edges = zip(keys_a[index_a], keys_b[index_b])

    molecule.add_edges_from(edges)


def add_inter_molecule_edges(molecules, edges):
    """
    Create edges between molecules.

    The function is given a list of molecules and a list of edges. Each edge is
    provided as a tuple of two nodes, each node being a tuple of the molecule
    index in the list of molecule, and the node key in that molecule. An edge
    therefore looks like ``((0, 10), (2, 20))`` where ``1`` and ``2`` are
    molecules, ``10`` is a key of ``molecules[0]``, and ``20`` is a key of
    ``molecules[2]``.

    The function **can** create edges within a molecule if the same molecule
    index is given for both ends of edges.

    Molecules that get linked are merged. In a merged molecule, the order of
    the input molecules is kept. In a list of molecules numbered from 0 to 4,
    if molecules 1, 2, and 4 are merged, then the result molecules are, in
    order, 0, 1-2-4, 3.

    Parameters
    ----------
    molecules: list
        List of molecules to link.
    edges: list
        List of edges in a ``(molecule_incex, node_key)`` format as described
        above.

    Returns
    -------
    list
        New list of molecules.
    """
    molecule_graph = nx.Graph()
    molecule_graph.add_nodes_from(range(len(molecules)))
    molecule_edges = [(edge[0][0], edge[1][0]) for edge in edges]
    molecule_graph.add_edges_from(molecule_edges)
    components = nx.connected_components(molecule_graph)

    # Do the merging
    new_molecules = []
    correspondance = {}
    for new_index, component in enumerate(components):
        # Connected components are yielded as sets. We need to be able to
        # iterate over them in numerical order.
        component = sorted(component)

        # The first molecule in the component is the base for the merge.
        base_index = component[0]
        base_molecule = molecules[base_index]
        new_molecules.append(base_molecule)
        for key in base_molecule:
            correspondance[(base_index, key)] = (new_index, key)
        offset = max(base_molecule.nodes)

        for other_index in component[1:]:
            other_molecule = molecules[other_index]
            mol_correspondance = base_molecule.merge_molecule(other_molecule)
            for before, after in mol_correspondance.items():
                correspondance[(other_index, before)] = (new_index, after)
            offset = max(base_molecule.nodes)
        print(new_molecules)

    # Create the edges
    for edge in edges:
        new_edge = (correspondance[edge[0]], correspondance[edge[1]])
        assert new_edge[0][0] == new_edge[1][0]
        molecule_index = new_edge[0][0]
        new_molecules[molecule_index].add_edge(new_edge[0][1], new_edge[1][1])

    return new_molecules


class RemoveCysteinBridgeEdges(Processor):
    def __init__(self, template=UNIVERSAL_BRIDGE_TEMPLATE):
        self.template = UNIVERSAL_BRIDGE_TEMPLATE

    def run_molecule(self, molecule):
        remove_cystein_bridge_edges(molecule, self.template)
        return molecule
