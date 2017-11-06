#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:51:03 2017

@author: peterkroon
"""

from .utils import maxes, first_alpha

import itertools
import networkx as nx


def categorical_cartesian_product(G, H, attributes=tuple()):
    P = nx.Graph()  # FIXME graphtype?
    for u, v in itertools.product(G, H):
        if all(G.node[u][attr] == H.node[v][attr] for attr in attributes):
            attrs = {}
            for attr in set(G.node[u].keys()) | set(H.node[v].keys()):
                attrs[attr] = (G.node[u].get(attr, None), H.node[v].get(attr, None))
            P.add_node((u, v), **attrs)
    return P


def categorical_modular_product(G, H, attributes=tuple()):
    P = categorical_cartesian_product(G, H, attributes)
    for (u1, v1), (u2, v2) in itertools.combinations(P.nodes(), 2):
        both_edge = G.has_edge(u1, u2) and H.has_edge(v1, v2)
        neither_edge = not G.has_edge(u1, u2) and not H.has_edge(v1, v2)
        # Effectively: not (G.has_edge(u1, u2) xor H.has_edge(v1, v2))
        if u1 != u2 and v1 != v2 and (both_edge or neither_edge):
            attrs = {}
            if both_edge:
                for attr in set(G.edges[u1, u2].keys()) | set(H.edges[v1, v2].keys()):
                    attrs[attr] = (G.edges[u1, u2].get(attr, None), H.edges[v1, v2].get(attr, None))
            P.add_edge((u1, v1), (u2, v2), **attrs)
    return P


def categorical_maximum_common_subgraph(G, H, attributes=tuple()):
    P = categorical_modular_product(G, H, attributes)
    cliques = nx.find_cliques(P)
    # cliques is an iterator which will return a *lot* of items. So make sure
    # we never turn it into a full list.
    largest = maxes(cliques, key=len)
    matches = [dict(clique) for clique in largest]
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
    matches : list of dictionaries
        The matches found. The dictionaries have node indices of ``reference`` as
        keys and node indices of ``residue`` as values. Is an empty list if
        ``residue`` is not a subgraph of ``reference``.
    """
    matches = []
#    H_idxs = [idx for idx in residue if residue.node[idx]['element'] == 'H']
    H_idxs = [idx for idx in residue if residue.degree(idx) == 1]
    heavy_res = residue.copy()
    heavy_res.remove_nodes_from(H_idxs)
    # First, generate all the isomorphisms on heavy atoms. For each of these
    # we'll find *something* where the hydrogens match.
    GM = ElementGraphMatcher(reference, heavy_res)
    first_matches = list(GM.subgraph_isomorphisms_iter())
    for match in first_matches:
        GM_large = ElementGraphMatcher(reference, residue)
        # Put the knowledge from the heavy atom isomorphism back in. Note that
        # ElementGraphMatched is modified to enable this and is no longer
        # re-entrant.
        # Indices in match do not have to be changed to account for interlaced
        # hydrogens: the node-indices in heavy_res and residue are the same.
        GM_large.core_1 = match
        GM_large.core_2 = {v: k for k, v in match.items()}
        outcome = GM_large.subgraph_isomorphisms_iter()
        # Take just the first match found, otherwise it becomes a combinatorics
        # problem (consider an alkane chain). This is fine though, since
        # hydrogrens are supposed to be equal. Let's say you have some sort of
        # chiral atom with two hydrogens: It's not chiral. Let's now say one of
        # the two is a deuterium: in that case you should have a proper
        # 'element' header, and the subgraph will be matched correctly.
        # TODO: test this
        # So worst case scenario we rename all hydrogens. This is acceptable
        # since they're equal.
        # And do islice since there may be none.
        matches.extend(itertools.islice(outcome, 1))
    matches = sorted(matches,
                     key=lambda m: rate_match(reference, residue, m),
                     reverse=True)
    return matches


class ElementGraphMatcher(nx.isomorphism.GraphMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().initialize()

    def initialize(self):
        return

    def semantic_feasibility(self, node1, node2):
        # TODO: implement (partial) wildcards
        elem1 = self.G1.node[node1].get('element', first_alpha(self.G1.node[node1]['atomname']))
        elem2 = self.G2.node[node2].get('element', first_alpha(self.G2.node[node2]['atomname']))
        return elem1 == elem2


def blockmodel(G, partitions, **attrs):
    """
    Analogous to networkx.blockmodel, but can deal with incomplete partitions,
    assigns ``attrs`` to nodes, and calculates the new ``position`` attribute as
    center of geometry.

    Parameters
    ----------
    G: networkx.Graph
        The graph to partition
    partitions: iterable of iterables
        Each element contains the node indices that construct the new node.
    **attrs: dict of str: iterable
        Attributes to assign to new nodes. Attribute values are assigned to the
        new nodes in order.

    Returns
    -------
    networkx.Graph
        A new graph where every node is a subgraph as specified by partitions.
        Node attributes:

            :graph: Subgraph of constructing nodes.
            :position: Center of geometry of the ``position`` of nodes in ``graph``.
            :nnodes: Number of nodes in ``graph``.
            :nedges: Number of edges in ``graph``.
            :density: Density of ``graph``.
            :attrs.keys(): As specified by ``**attrs``.
    """
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
    return sum(residue.node[rdx]['atomname'] == bead.node[bdx]['atomname']
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
            :position: The center of geometry of the ``position`` of the nodes in
                       ``graph``.
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
    chain, resids, resnames = map(list, zip(*keys))
    res_graph = blockmodel(mol, grps, chain=chain, resid=resids,
                           resname=resnames, atomname=resnames)
    return res_graph

