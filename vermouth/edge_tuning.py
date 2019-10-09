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

"""
Set of tools to add and remove edges.
"""

import functools
import numpy as np
import networkx as nx

from . import KDTree
from . import selectors
from . import geometry


def _edge_is_between_selections(edge, selection_a, selection_b):
    """
    Returns ``True`` is the edge has one end in each selection.

    Parameters
    ----------
    edge: tuple[int, int]
    selection_a: collections.abc.Container[collections.abc.Hashable]
    selection_b: collections.abc.Container[collections.abc.Hashable]

    Returns
    -------
    bool
    """
    return (
        (edge[0] in selection_a and edge[1] in selection_b)
        or (edge[1] in selection_a and edge[0] in selection_b)
    )


def prune_edges_between_selections(molecule, selection_a, selection_b):
    """
    Remove edges which have their ends part of given selections.

    An edge is removed if has one end that is part of 'selection_a', and the
    other end part of 'selection_b'.

    Parameters
    ----------
    molecule: networkx.Graph
        Molecule to prune in-place.
    selection_a: collections.abc.Iterable[collections.abc.Hashable]
        List of node keys from the molecule.
    selection_b: collections.abc.Iterable[collections.abc.Hashable]
        List of node keys from the molecule.

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
        if _edge_is_between_selections(edge, selection_a, selection_b)
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
    molecule: networkx.Graph
        Molecule to prune in-place.
    selector_a: collections.abc.Callable
        A selector for one end of the edges.
    selector_b: collections.abc.Callable
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


def add_edges_at_distance(molecule, threshold,
                          selection_a, selection_b, attribute='position'):
    """
    Add edges within a molecule when the distance is below a threshold.

    Create edges within a molecule between nodes that have an end part of
    'selection_a', the other end part of 'selection_b', and a distance between
    the ends that is lesser than the given threshold.

    All nodes that are part of 'selection_a' or 'selection_b' must have a
    position stored under the attribute which key is given with the 'attribute'
    argument. That key is 'position' by default. If at least one node has the
    position missing, then a :exc:`KeyError` is raised.

    Parameters
    ----------
    molecule: networkx.Graph
        Molecule to modify in-place.
    threshold: float
        The distance threshold under which edges will be created. The distance
        is expressed in nm.
    selection_a: collections.abc.Iterable[collections.abc.Hashable]
        List of node keys from the molecule.
    selection_b: collections.abc.Iterable[collections.abc.Hashable]
        List of node keys from the molecule.
    attribute: collections.abc.Hashable
        Name of the key in the node dictionaries under which the coordinates
        are stored.

    Raises
    ------
    KeyError
        At least one node from the selections does not have a position.
    """
    selection_a = set(selection_a)
    selection_b = set(selection_b)
    keys_a = np.array([key for key in molecule.nodes.keys() if key in selection_a])
    keys_b = np.array([key for key in molecule.nodes.keys() if key in selection_b])
    coordinates_a = np.stack([
        molecule.nodes[key][attribute] for key in keys_a
    ])
    coordinates_b = np.stack([
        molecule.nodes[key][attribute] for key in keys_b
    ])

    distance_matrix = geometry.distance_matrix(coordinates_a, coordinates_b)
    index_a, index_b = np.where(distance_matrix < threshold)
    edges = (
        (node1, node2, {'distance': distance})
        for node1, node2, distance
        in zip(keys_a[index_a], keys_b[index_b], distance_matrix[index_a, index_b])
    )

    molecule.add_edges_from(edges)


def add_inter_molecule_edges(molecules, edges):
    """
    Create edges between molecules.

    The function is given a list of molecules and a list of edges. Each edge is
    provided as a tuple of two nodes, each node being a tuple of the molecule
    index in the list of molecule, and the node key in that molecule. An edge
    therefore looks like ``((0, 10), (2, 20))`` where ``1`` and ``2`` are
    indices of molecules in `molecules`, ``10`` is the key of a node from
    ``molecules[0]``, and ``20`` is the key of a node from ``molecules[2]``.

    The function **can** create edges within a molecule if the same molecule
    index is given for both ends of edges.

    Molecules that get linked are merged. In a merged molecule, the order of
    the input molecules is kept. In a list of molecules numbered from 0 to 4,
    if molecules 1, 2, and 4 are merged, then the result molecules are, in
    order, 0, 1-2-4, 3.

    Parameters
    ----------
    molecules: collections.abc.Sequence[vermouth.molecule.Molecule]
        List of molecules to link.
    edges: collections.abc.Iterable[tuple[int, collections.abc.Hashable]]
        List of edges in a ``(molecule_index, node_key)`` format as described
        above. Edges can have a third element, it is then a dictionary of
        attributes to be attached to the edge.

    Returns
    -------
    list
        New list of molecules.
    """
    edges = tuple(edges)  # We iterate multiple time trhough it.
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

        for other_index in component[1:]:
            other_molecule = molecules[other_index]
            mol_correspondance = base_molecule.merge_molecule(other_molecule)
            for before, after in mol_correspondance.items():
                correspondance[(other_index, before)] = (new_index, after)

    # Create the edges
    for edge in edges:
        new_edge = (correspondance[edge[0]], correspondance[edge[1]])
        assert new_edge[0][0] == new_edge[1][0], \
            'The two ends of the edge are not part of the same molecule.'
        molecule_index = new_edge[0][0]
        new_molecules[molecule_index].add_edge(new_edge[0][1], new_edge[1][1])
        if len(edge) > 2:
            in_mol_edge = (
                new_molecules[molecule_index]
                .edges[new_edge[0][1], new_edge[1][1]]
            )
            in_mol_edge.update(edge[2])

    return new_molecules


def pairs_under_threshold(molecules, threshold,
                          selection_a, selection_b,
                          attribute='position', min_edges=0):
    """
    List pairs of nodes from a selection that are closer than a threshold.

    Get the distance between nodes from multiple molecules and list the pairs
    that are closer than the given threshold. The molecules are given as a list
    of molecules, the selection is a list of nodes each of them a tuple
    ``(index of the molecule in the list, key of the node in the molecule)``.
    The result of the function is a generator of node pairs followed by the
    distance between the nodes, each node formated as in the selection.

    All nodes from the selection must have a position accessible under the key
    given as the 'attribute' argument. That key is 'position' by default.

    With the `min_edges` argument, one can prevent pairs to be selected if
    there is a path between two nodes that is shorter than a given number of
    edges.

    Parameters
    ----------
    molecules: collections.abc.Collection[vermouth.molecule.Molecule]
        A list of :class:`vermouth.molecule.Molecule`.
    threshold: float
        A distance threshold in nm. Pairs are return if the nodes are closer
        than this threshold.
    selection_a: collections.abc.Iterable[collections.abc.Hashable]
        List of nodes to consider at one end of the pairs. The format is
        described above.
    selection_b: collections.abc.Iterable[collections.abc.Hashable]
        List of nodes to consider at the other end of the pairs. The format is
        described above.
    attribute: collections.abc.Hashable
        The dictionary key under which the node positions are stored in the
        nodes.
    min_edges: int
        Do not select pairs that are connected by less than that number of
        edges.

    Yields
    ------
    tuple[collections.abc.Hashable, collections.abc.Hashable, float]
        Pairs of node closer than the threshold in the format described above
        and the distance between the nodes.

    Raises
    ------
    KeyError
        Raised if a node from the selection does not have a position.

    Notes
    -----

    Symetric node pairs are not deduplicated.
    """
    coordinates_a = []
    for key in selection_a:
        coordinates_a.append(molecules[key[0]].nodes[key[1]][attribute])
    coordinates_b = []
    for key in selection_b:
        coordinates_b.append(molecules[key[0]].nodes[key[1]][attribute])
    if not coordinates_a or not coordinates_b:
        return
    kdtree_a = KDTree(coordinates_a)
    kdtree_b = KDTree(coordinates_b)
    sparse_distance_matrix = kdtree_a.sparse_distance_matrix(kdtree_b, threshold)
    for (idx, jdx), distance_between in sparse_distance_matrix.items():
        key_a = selection_a[idx]
        key_b = selection_b[jdx]
        if key_a != key_b and distance_between < threshold:
            if not _are_close(min_edges, molecules, key_a, key_b):
                yield (key_a, key_b, distance_between)


def _are_close(threshold, molecules, key_a, key_b):
    if not threshold:
        return False
    if key_a[0] != key_b[0]:
        return False
    try:
        length = nx.shortest_path_length(
            molecules[key_a[0]], key_a[1], key_b[1]
        )
    except nx.exception.NetworkXNoPath:
        return False
    else:
        return length < threshold


def select_nodes_multi(molecules, selector):
    """
    Find the nodes that correspond to a selector among multiple molecules.

    Runs a selector over multiple molecules. The selector must be a function
    that takes a node dictionary as argument and returns ``True`` if the node
    should be selected. The selection is yielded as tuples of a molecule indice
    from the molecule list input, and a key from the molecule.

    Parameters
    ----------
    molecule: collections.abc.Iterable[Molecule]
        A list of molecules.
    selector: collections.abc.Callable
        A selector function.

    Yields
    ------
    tuple[int, collections.abc.Hashable]
        Molecule/key identifier for the selected nodes.
    """
    for molecule_idx, molecule in enumerate(molecules):
        for key, node in molecule.nodes.items():
            if selector(node):
                yield (molecule_idx, key)


def add_edges_threshold(molecules, threshold,
                        templates_a, templates_b,
                        attribute='position', min_edges=0):
    """
    Add edges between two selections when under a given threshold.

    Edges are added within and between the molecules and connect nodes that
    match the given template. Molecules that get connected by an edge are
    merged and the new list of molecules is returned.

    Parameters
    ----------
    molecules: collections.abc.Sequence[Molecule]
        A list of molecules.
    threshold: float
        The distance threshold in nanometers under which an edge is created.
    templates_a: dict
        A list of templates; a node need to match at least one of them to be
        selected at one end.
    templates_b: dict
        A list of templates; a node need to match at least one of them to be
        selected at the other end.
    attribute: str
        Name of the key in the node dictionaries under which the coordinates
        are stored.
    min_edges: int
        Minimum number of edges between to nodes for an edge to be added.

    Returns
    -------
    list[vermouth.molecule.Molecule]
        A new list of molecules.
    """
    selector_a = functools.partial(
        selectors.proto_multi_templates, templates=templates_a
    )
    selection_a = list(select_nodes_multi(molecules, selector_a))
    selector_b = functools.partial(
        selectors.proto_multi_templates, templates=templates_b
    )
    selection_b = list(select_nodes_multi(molecules, selector_b))
    edges = pairs_under_threshold(molecules, threshold,
                                  selection_a, selection_b,
                                  attribute, min_edges=min_edges)
    edges = (
        (node1, node2, {'distance': distance})
        for node1, node2, distance in edges
    )
    new_molecules = add_inter_molecule_edges(molecules, edges)
    return new_molecules
