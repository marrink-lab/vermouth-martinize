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

from collections import defaultdict
import itertools
import networkx as nx

from .utils import maxes, first_alpha, are_all_equal


def add_element_attr(molecule):
    """
    Adds an element attribute to every node in `molecule`, based on that node's
    atomname attribute.

    Parameters
    ----------
    molecule: networkx.Graph
        The graph of which nodes should get an element attribute.

    Raises
    ------
    ValueError
        If no element could be guessed for a node.

    """
    for node_idx in molecule:
        node = molecule.nodes[node_idx]
        if 'element' not in node:
            try:
                element = first_alpha(node['atomname'])
            except KeyError:
                raise ValueError('Cannot guess the element of atom {}: '
                                 'the node has no atom name.'
                                 .format(node_idx))
            except ValueError:
                raise ValueError('Cannot guess the element of atom {}: '
                                 'the atom name has no alphabetic charater.'
                                 .format(node_idx))
            node['element'] = element


def categorical_cartesian_product(graph1, graph2, attributes=tuple()):
    product = nx.Graph()  # FIXME graphtype?
    for idx1, idx2 in itertools.product(graph1, graph2):
        node1 = graph1.nodes[idx1]
        node2 = graph2.nodes[idx2]
        if all((attr in node1 and attr in node2 and node1[attr] == node2[attr])
               or (attr not in node1 and attr not in node2) for attr in attributes):
            attrs = {}
            for attr in set(node1.keys()) | set(node2.keys()):
                attrs[attr] = (node1.get(attr, None), node2.get(attr, None))
            product.add_node((idx1, idx2), **attrs)
    return product


def categorical_modular_product(graph1, graph2, attributes=tuple()):
    product = categorical_cartesian_product(graph1, graph2, attributes)
    for (graph1_node1, graph2_node1), (graph1_node2, graph2_node2) in itertools.combinations(product.nodes(), 2):
        graph1_nodes = graph1_node1, graph1_node2
        graph2_nodes = graph2_node1, graph2_node2
        both_edge = graph1.has_edge(*graph1_nodes) and graph2.has_edge(*graph2_nodes)
        neither_edge = not graph1.has_edge(*graph1_nodes) and\
                       not graph2.has_edge(*graph2_nodes)
        # Effectively: not (graph1.has_edge(graph1_node1, graph1_node2) xor graph2.has_edge(graph2_node1, graph2_node2))
        if graph1_node1 != graph1_node2 and graph2_node1 != graph2_node2 and\
                (both_edge or neither_edge):
            attrs = {}
            if both_edge:
                g1_edge_keys = set(graph1.edges[graph1_nodes].keys())
                g2_edge_keys = set(graph2.edges[graph2_nodes].keys())
                for attr in g1_edge_keys | g2_edge_keys:
                    attrs[attr] = (graph1.edges[graph1_nodes].get(attr, None),
                                   graph2.edges[graph2_nodes].get(attr, None))
            product.add_edge((graph1_node1, graph2_node1), (graph1_node2, graph2_node2), **attrs)
    return product


def categorical_maximum_common_subgraph(graph1, graph2, attributes=tuple()):
    product = categorical_modular_product(graph1, graph2, attributes)
    cliques = nx.find_cliques(product)
    # cliques is an iterator which will return a *lot* of items. So make sure
    # we never turn it into a full list.
    largest = maxes(cliques, key=len)
    matches = [dict(clique) for clique in largest]
    return matches


def rate_match(residue, bead, match):
    """
    A helper function which rates how well ``match`` describes the isomorphism
    between ``residue`` and ``bead`` based on the number of matching atomnames.


    Parameters
    ----------
    residue : networkx.Graph
        A graph. Required node attributes:

            :atomname: The name of an atom.

    bead : networkx.Graph
        A subgraph of ``residue`` where the isomorphism is described by ``match``.
        Required node attributes:

            :atomname: The name of an atom.


    Returns
    -------
    int
        The number of entries in match where the atomname in ``residue`` matches
        the atomname in ``bead``.
    """
    return sum(residue.nodes[rdx].get('atomname') == bead.nodes[bdx].get('atomname')
               for rdx, bdx in match.items())


def _items_with_common_values(graph, nodes=None, excluded_keys=[]):
    """
    Finds all node attributes of nodes in graph that all have the same values.
    Returns a dict of node attribute/common value pairs. One can exclude specifc
    keys using the exclude variable.

    Parameters
    ----------
    graph: networkx.Graph
    nodes: collections.abc.Iterable or None
        If None, consider all nodes. All nodes must be in graph.
    exclude:  collections.abc.Iterable
        keys to exclude from the common value list

    Returns
    -------
    dict
    """
    if not nodes:
        nodes = set(graph.nodes)

    common_attrs = defaultdict(list)
    for idx in nodes:
        for key, val in graph.nodes[idx].items():
            if key not in excluded_keys:
                common_attrs[key].append(val)
    common_attrs = {key: vals[0] for key, vals in common_attrs.items()
                    if len(vals) == len(nodes) and are_all_equal(vals)}
    return common_attrs


def get_attrs(node, attrs):
    """
    Returns multiple values from a dictionary in order.

    Parameters
    ----------
    node: dict
        The dict from which items should be taken.
    attrs: collections.abc.Iterable
        The keys which values should be taken.

    Returns
    -------
    tuple
        A tuple containing the value of every key in attrs in the same order,
        where missing values are `None`.
    """
    return tuple(node.get(attr) for attr in attrs)


def partition_graph(graph, partitions):
    """
    Create a new graph based on `graph`, where nodes are aggregated based on
    `partitions`, similar to :func:`~networkx.algorithms.minors.quotient_graph`,
    except that it only accepts pre-made partitions, and edges are not given
    a 'weight' attribute. Much fast than the quotient_graph, since it creates
    edges based on existing edges rather than trying all possible combinations.

    Parameters
    ----------
    graph: networkx.Graph
        The graph to partition
    partitions: collections.abc.Iterable[collections.abc.Iterable[collections.abc.Hashable]]
        E.g. a list of lists of node indices, describing the partitions. Will
        be sorted by lowest index.

    Returns
    -------
    networkx.Graph
        The coarser graph.
    """
    new_graph = nx.Graph()
    partitions = sorted(partitions, key=min)
    mapping = {}
    for idx, node_idxs in enumerate(partitions):
        subgraph = nx.subgraph(graph, node_idxs)
        new_graph.add_node(idx,
                           graph=subgraph,
                           nnodes=len(subgraph),
                           nedges=len(subgraph.edges),
                           density=nx.density(subgraph))
        mapping.update({node_idx: idx for node_idx in node_idxs})

    for idx, jdx in graph.edges:
        if mapping[idx] != mapping[jdx]:
            new_idx, new_jdx = mapping[idx], mapping[jdx]
            edge_attrs = graph.edges[(idx, jdx)]
            if new_graph.has_edge(new_idx, new_jdx):
                old_attrs = new_graph.edges[(new_idx, new_jdx)]
                new_attrs = {key: old_attrs[key] for key in old_attrs.keys() & edge_attrs.keys()\
                                                         if old_attrs[key] == edge_attrs[key]}
                old_attrs.clear()
                new_graph.add_edge(new_idx, new_jdx, **new_attrs)
            else:
                new_graph.add_edge(new_idx, new_jdx, **edge_attrs)
    return new_graph

def make_residue_graph(graph, attrs=('chain', 'resid', 'resname', 'insertion_code')):
    """
    Create a new graph based on `graph`, where nodes with identical attribute
    values for the attribute names in `attrs` will be contracted into a single,
    coarser node. With the default arguments it will create a graph with one
    node per residue.
    Resulting (coarse) nodes will have the same attributes as the constructing
    nodes, but only those that have identical values. In addition, they'll have
    attributes 'graph', 'nnodes', 'nedges' and 'density'.

    Parameters
    ----------
    graph: networkx.Graph
        The graph to condense.
    attrs: collections.abc.Iterable[collections.abc.Hashable]
        The node attributes that determine node equivalence.

    Returns
    -------
    networkx.Graph
        The resulting coarser graph, where equivalent nodes are contracted to a
        single node.
    """
    # Create partitions. These will contain all nodes, even those without e.g.
    # a resname, since those will get resname None
    residue_idxs = collect_residues(graph, attrs)
    res_graph = partition_graph(graph, residue_idxs.values())
    # Alternatively we can use nx.quotient_graph, but that slows down the
    # program for large-ish molecules, since quotient_graph scales as O(N^2).
    # Our partition_graph scales much better, as O(E*N) (number of edges, nodes)
    # res_graph = nx.quotient_graph(graph,
    #                               sorted(residue_idxs.values(), key=min),
    #                               relabel=True)
    # Using this equivalence function rather than the preformed partitions
    # Creates an equivalent graph, but the node indices are numbered
    # differently. At the very least it would require a change in the tests.
    # Note2: This would probably break e.g. the DSSP processor, because the
    # ordering of residues in the molecule comes from here, and is used by
    # molecule.iter_residues.
    # def node_equiv(idx, jdx):
    #     return get_attrs(graph.nodes[idx], attrs) == get_attrs(graph.nodes[jdx], attrs)
    # res_graph = nx.quotient_graph(graph, node_equiv, relabel=True)
    for res_idx in res_graph:
        res_node = res_graph.nodes[res_idx]
        res_node.update(_items_with_common_values(res_node['graph'], excluded_keys=['graph']))
    return res_graph


def collect_residues(graph, attrs=('chain', 'resid', 'resname', 'insertion_code')):
    """
    Creates groups of indices based on the node attributes with keys `attrs`.
    All nodes in graph will be part of exactly one group.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        The graph whose node indices should be grouped.
    attrs: :class:`~collections.abc.Sequence`
        The attribute keys that should be used to group node indices. The
        associated values should be hashable.

    Returns
    -------
    dict[tuple, set]
        The keys are the found node attributes, the values the associated node
        indices.
    """
    residues = defaultdict(set)
    for node_idx in graph:
        key = get_attrs(graph.nodes[node_idx], attrs=attrs)
        residues[key].add(node_idx)
    return dict(residues)


# We can't inherit from nx.isomorphism.GraphMatcher to override
# `semantic_feasibility`. That implementation will clobber this one's method.
class MappingGraphMatcher(nx.isomorphism.isomorphvf2.GraphMatcher):
    def __init__(self, *args, edge_match=None, node_match=None, **kwargs):
        self.edge_match = edge_match
        self.node_match = node_match
        super().__init__(*args, **kwargs)
        self.G1_adj = self.G1.adj

    def semantic_feasibility(self, G1_node, G2_node):
        """
        Returns True if mapping G1_node to G2_node is semantically feasible.
        Adapted from networkx.algorithms.isomorphism.vf2userfunc._semantic_feasibility.
        """
        # Make sure the nodes match
        if self.node_match is not None:
            nm = self.node_match(self.G1.nodes[G1_node], self.G2.nodes[G2_node])
            if not nm:
                return False

        # Make sure the edges match
        if self.edge_match is not None:

            # Cached lookups
            core_1 = self.core_1
            edge_match = self.edge_match

            for neighbor in self.G1_adj[G1_node]:
                # G1_node is not in core_1, so we must handle R_self separately
                if neighbor == G1_node:
                    if not edge_match(G1_node, G1_node, G2_node, G2_node):
                        return False
                elif neighbor in core_1:
                    if not edge_match(G1_node, neighbor, G2_node, core_1[neighbor]):
                        return False
            # syntactic check has already verified that neighbors are symmetric
        return True
