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

import itertools
import networkx as nx

from .ismags import ISMAGS
from .utils import maxes, first_alpha


def add_element_attr(molecule):
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


def blockmodel(G, partitions, **attrs):
    """
    Analogous to networkx.blockmodel, but can deal with incomplete partitions,
    and assigns ``attrs`` to nodes.

    Parameters
    ----------
    G: networkx.Graph
        The graph to partition
    parititions: collections.abc.Iterable[collections.abc.Iterable]
        Each element contains the node indices that construct the new node.
    **attrs: dict[str, collections.abc.Iterable]
        Attributes to assign to new nodes. Attribute values are assigned to the
        new nodes in order.

    Returns
    -------
    networkx.Graph
        A new graph where every node is a subgraph as specified by partitions.
        Node attributes:

            :graph: Subgraph of constructing nodes.
            :nnodes: Number of nodes in ``graph``.
            :nedges: Number of edges in ``graph``.
            :density: Density of ``graph``.
            :attrs.keys(): As specified by ``**attrs``.
    """
    # TODO: Change this to use nx.quotient_graph.
    attrs = {key: list(val) for key, val in attrs.items()}
    CG_mol = nx.Graph()
    for bead_idx, idxs in enumerate(partitions):
        bd = G.subgraph(idxs)
        CG_mol.add_node(bead_idx)
        CG_mol.node[bead_idx]['graph'] = bd
        # TODO: CoM instead of CoG
#        CG_mol.node[bead_idx]['position'] = np.mean([bd.node[idx]['position'] for idx in bd], axis=0)
        for k, vals in attrs.items():
            CG_mol.node[bead_idx][k] = vals[bead_idx]

        CG_mol.node[bead_idx]['nnodes'] = bd.number_of_nodes()
        CG_mol.node[bead_idx]['nedges'] = bd.number_of_edges()
        CG_mol.node[bead_idx]['density'] = nx.density(bd)

    block_mapping = {}
    for n in CG_mol:
        nodes_in_block = CG_mol.node[n]['graph'].nodes()
        block_mapping.update(dict.fromkeys(nodes_in_block, n))

    for u, v, d in G.edges(data=True):
        try:
            bmu = block_mapping[u]
            bmv = block_mapping[v]
        except KeyError:
            # Atom not represented
            continue
        if bmu == bmv:  # no self loops
            continue
        # For graphs and digraphs add single weighted edge
        weight = d.get('weight', 1.0)  # default to 1 if no weight specified
        if CG_mol.has_edge(bmu, bmv):
            CG_mol[bmu][bmv]['weight'] += weight
        else:
            CG_mol.add_edge(bmu, bmv, weight=weight)
    return CG_mol


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
    return sum(residue.node[rdx].get('atomname') == bead.node[bdx].get('atomname')
               for rdx, bdx in match.items())


def make_residue_graph(mol):
    """
    Creates a graph with one node per residue; as identified by the tuple
    (chain identifier, residue index, residue name).

    Parameters
    ----------
    mol: networkx.Graph
        The atomistic graph. Required node attributes:

            :chain: The chain identifier.
            :resid: The residue index.
            :resname: The residue name.

    Returns
    -------
    networkx.Graph
        A graph with one node per residue. Node attributes:

            :chain: The chain identifier.
            :graph: The atomistic subgraph.
            :density: The density of ``graph``.
            :nedges: The number of edges in ``graph``.
            :nnodes: The number of nodes in ``graph``.
            :resid: The residue index.
            :resname: The residue name.
            :atomname: The residue name.
    """
    def keyfunc(node_idx):
        return mol.node[node_idx]['chain'], mol.node[node_idx]['resid'], mol.node[node_idx]['resname']
    nodes = sorted(mol.node, key=keyfunc)
    keys = []
    grps = []
    for key, grp in itertools.groupby(nodes, keyfunc):
        keys.append(key)
        grps.append(list(grp))
    if keys:
        chain, resids, resnames = map(list, zip(*keys))
    else:
        chain, resids, resnames = [], [], []
    res_graph = blockmodel(mol, grps, chain=chain, resid=resids,
                           resname=resnames, atomname=resnames)
    return res_graph
