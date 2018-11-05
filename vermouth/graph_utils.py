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


def maximum_common_subgraph(graph1, graph2, attributes=tuple()):
    product = nx.Graph()
    # First, find the MCS between all nodes of degree != 1, such as the carbons
    # Nothing new or exciting here.
    for g1_node, g2_node in itertools.product(graph1, graph2):
        node1 = graph1.nodes[g1_node]
        node2 = graph2.nodes[g2_node]
        if all((attr in node1 and attr in node2 and node1[attr] == node2[attr])
               or (attr not in node1 and attr not in node2) for attr in attributes):
            if graph1.degree(g1_node) != 1 and graph2.degree(g2_node) != 1:
                product.add_node((g1_node, g2_node))
    for (g1_node1, g2_node1), (g1_node2, g2_node2) in itertools.combinations(product.nodes(), 2):
        both_edge = graph1.has_edge(g1_node1, g1_node2) and graph2.has_edge(g2_node1, g2_node2)
        neither_edge = not graph1.has_edge(g1_node1, g1_node2) and not graph2.has_edge(g2_node1, g2_node2)
        # Effectively: not (graph1.has_edge(g1_node1, g1_node2) xor graph2.has_edge(g2_node1, g2_node2))
        if g1_node1 != g1_node2 and g2_node1 != g2_node2 and (both_edge or neither_edge):
            product.add_edge((g1_node1, g2_node1), (g1_node2, g2_node2))
    cliques = nx.find_cliques(product)
#    largest = maxes(cliques, key=len)
    # We can't do maxes, because it might still grow to be as large. Does make
    # things slower though... We could say that it can grow to be at most the
    # current size plus the number of degree-1 nodes.
    largest = cliques

    # Add an empty match in case nothing of degree > 1 matches. In that case we
    # still need to do the loop below.
    largest = itertools.chain([[]], largest)

    # Now, for every MCS we found, look at the nodes of degree 1. The
    # attributes still need to match. In addition, they need to have the same
    # (mapped) neighbour, or the neighbour must be missing from the graph2 graph
    all_cliques = []
    for clique in largest:
        match = dict(clique)
        product = nx.Graph()
        product.add_nodes_from(clique)
        for g1_node, g2_node in itertools.product(graph1, graph2):
            node1 = graph1.nodes[g1_node]
            node2 = graph2.nodes[g2_node]
            # We can't do this above, because we need the match to translate
            # nodes from graph graph1 to graph graph2 to see whether their neighbours
            # correspond.
            if (graph1.degree(g1_node) <= 1 or graph2.degree(g2_node) <= 1) and\
                    all((attr in node1 and attr in node2 and node1[attr] == node2[attr])
                            or (attr not in node1 and attr not in node2) for attr in attributes):
                g1_neighbors = [match.get(n, None) for n in graph1.neighbors(g1_node)]
                # If no neighbors are found for g1_node, or if any of them are
                # the same in graph2, they're compatible.

                # FIXME?
                # This eliminates some nodes from the MCS, since it's possible
                # the MCS does not include *all* nodes in match. This means
                # that some nodes should be considered compatible, even if they
                # have different neighbours, but only if that neighbor is not
                # part of the final MCS. This makes testing a little tricky,
                # since categorical_maximum_common_subgraph *does* find them.
                # It'll be a cornercase anyway.
                if not g1_neighbors or None in g1_neighbors or any(n in g1_neighbors for n in graph2.neighbors(g2_node)):
                    product.add_node((g1_node, g2_node))
        for (g1_node1, g2_node1), (g1_node2, g2_node2) in itertools.combinations(product.nodes(), 2):
            both_edge = graph1.has_edge(g1_node1, g1_node2) and graph2.has_edge(g2_node1, g2_node2)
            neither_edge = not graph1.has_edge(g1_node1, g1_node2) and not graph2.has_edge(g2_node1, g2_node2)
            # Effectively: not (graph1.has_edge(g1_node1, g1_node2) xor graph2.has_edge(g2_node1, g2_node2))
            if g1_node1 != g1_node2 and g2_node1 != g2_node2 and (both_edge or neither_edge):
                product.add_edge((g1_node1, g2_node1), (g1_node2, g2_node2))
        # TODO: This duplicates a lot of effort. Maybe create the compatibility
        # graph first from all cliques, and find the cliques only once?
        this_pass = nx.find_cliques(product)
        all_cliques.append(this_pass)
    # And finally, find the largest MCS in all cliques graph2.
    largest = maxes(itertools.chain(*all_cliques), key=len)
    matches = set(frozenset(m) for m in largest)  # remove duplicates
    matches = [dict(clique) for clique in matches]

    return matches


def isomorphism(reference, residue):
    """
    Finds matching atoms between ``reference`` and ``residue``. ``residue`` should be
    a subgraph of ``reference``. Matchin is done based on connectivity and
    the ``element`` attribute of the nodes.

    The subgraph isomorphism is first calculated using non-hydrogen atoms only.
    These matches are then extended to include one option for the hydrogen
    isomorphism. This is done because otherwise a combinatorial problem is
    created: take for example an alkane chain: the carbon atoms match in one
    way. Then, for every :math:``CH_2`` group there are two, independent options
    for the hydrogen isomorphism. This would result in :math:``2^n`` subgraph
    isomorphisms for :math:``n`` carbon atoms.

    This means that the matches found will not be optimal for the hydrogens.
    This is acceptable, since hydrogrens are supposed to be equal. Let's say
    you have some sort of chiral atom with two hydrogens: it's not chiral and
    the hydrogen atoms are equal. Let's now say one of the two is a deuterium:
    in that case you should have a proper 'element' header, and the subgraph
    will be matched correctly.

    Parameters
    ----------
    reference : networkx.Graph
        The reference graph.
    residue : networkx.Graph
        The graph to match to ``reference``.
    Returns
    -------
    matches : list[dict]
        The matches found. The dictionaries have node indices of ``reference`` as
        keys and node indices of ``residue`` as values. Is an empty list if
        ``residue`` is not a subgraph of ``reference``.
    """

    new_residue_names = {name: idx for idx, name in enumerate(sorted(residue, key=lambda jdx: residue.nodes[jdx].get('atomname', '')))}
    new_reference_names = {name: idx for idx, name in enumerate(sorted(reference, key=lambda jdx: reference.nodes[jdx].get('atomname', '')))}
    old_res_names = {v: k for k, v in new_residue_names.items()}
    old_ref_names = {v: k for k, v in new_reference_names.items()}

    node_matcher = nx.isomorphism.categorical_node_match('element', None)

    reference = nx.relabel_nodes(reference, new_reference_names, copy=True)
    residue = nx.relabel_nodes(residue, new_residue_names, copy=True)

    ism = ISMAGS(reference, residue, node_match=node_matcher)
    matches = ism.find_subgraphs(symmetry=True)
    matches = sorted(matches,
                     key=lambda m: rate_match(reference, residue, m),
                     reverse=True)
    out = []
    for match in matches:
        out.append({old_ref_names[ref]: old_res_names[res] for ref, res in match.items()})
    return out


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
