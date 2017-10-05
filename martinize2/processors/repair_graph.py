#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:43:19 2017

@author: peterkroon
"""

from .processor import Processor
from ..utils import first_alpha, maxes

import functools
import itertools
import os.path

import networkx as nx


try:
    import pkg_resources
    DATA_PATH = pkg_resources.resource_filename('martinize2', 'mapping')
except ImportError:
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'mapping')


def categorical_cartesian_product(G, H, attributes=tuple()):
    P = nx.Graph()  # FIXME graphtype
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
    H_idxs = [idx for idx in residue if residue.node[idx]['element'] == 'H']
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


@functools.lru_cache(None)
def read_reference_graph(resname):
    """
    Reads the reference graph from ./mapping/universal/{resname},gml.

    Parameters
    ----------
    resname : str
        The residuename as found in the PDB file.

    Returns
    -------
    networkx.Graph
        Reference graph of the residue.
    """
    return nx.read_gml(os.path.join(DATA_PATH, 'universal', '{}.gml'.format(resname)), label='id')
#    return nx.read_gml('/universal/{}.gml'.format(resname), label='id')


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


def add_element_attr(molecule):
    for node_idx in molecule:
        node = molecule.node[node_idx]
        node['element'] = node.get('element', first_alpha(node['atomname']))


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


def make_reference(mol):
    """
    Takes an atomistic reference graph as read from a PDB file, and finds and
    returns the graph how it should look like, including all matching nodes
    between the input graph and the references.
    Requires residuenames to be correct.

    Notes
    -----
        The match between hydrogren atoms need not be perfect. See the
        documentation of ``isomorphism``.

    Parameters
    ----------
    mol : networkx.Graph
        The graph read from e.g. a PDB file. Required node attributes:

        :resname: The residue name.
        :resid: The residue id.
        :chain: The chain identifier.
        :element: The element.
        :atomname: The atomname.

    Returns
    -------
    networkx.Graph
        The constructed reference graph with the following node attributes:

        :resid: The residue id.
        :resname: The residue name.
        :chain: The chain identifier.
        :found: The residue subgraph from the PDB file.
        :reference: The residue subgraph used as reference.
        :match: A dictionary describing how the reference corresponds
            with the provided graph. Keys are node indices of the
            reference, values are node indices of the provided graph.
    """
    reference_graph = nx.Graph()
    residues = make_residue_graph(mol)

    for residx in residues:
        # TODO: make separate function for just one residue.
        # TODO: multiprocess this loop?
        # TODO: Merge degree 1 nodes (hydrogens!) with the parent node. And
        # check whether the node degrees match.

        resname = residues.node[residx]['resname']
        resid = residues.node[residx]['resid']
        chain = residues.node[residx]['chain']
        # print("{}{}".format(resname, resid), flush=True)
        residue = residues.node[residx]['graph']
        reference = read_reference_graph(resname)
        add_element_attr(reference)
        add_element_attr(residue)
        # Assume reference >= residue
        matches = isomorphism(reference, residue)
        if not matches:
            print('Doing MCS matching for residue {}{}'.format(resname, resid))
            # The problem is that some residues (termini in particular) will
            # contain more atoms than they should according to the reference.
            # Furthermore they will have too little atoms because X-Ray is
            # supposedly hard. This means we can't do the subgraph isomorphism like
            # we're used to. Instead, identify the atoms in the largest common
            # subgraph, and do the subgraph isomorphism/alignment on those. MCS is
            # ridiculously expensive, so we only do it when we have to.
            try:
                mcs_match = max(categorical_maximum_common_subgraph(reference, residue, ['element']),
                                key=lambda m: rate_match(reference, residue, m))
            except ValueError:
                raise ValueError('No common subgraph found between {} and reference {}'.format(resname, resname))
            # We could seed the isomorphism calculation with the knowledge from the
            # mcs_match, but thats to much effort for now.
            # TODO: see above
            res = residue.subgraph(mcs_match.values())
            matches = isomorphism(reference, res)
        match = matches[0]
        reference_graph.add_node(residx, chain=chain, reference=reference, found=residue, resname=resname, resid=resid, match=match)
    reference_graph.add_edges_from(residues.edges())
    return reference_graph


def repair_graph(aa_graph, reference_graph):
    """
    Repairs a graph ``aa_graph`` produced from a PDB file based on the
    information in ``reference_graph``. Missing atoms will be reconstructed and
    atom- and residue names will be canonicalized.

    Parameters
    ----------
    aa_graph : networkx.Graph
        The graph read from e.g. a PDB file. Required node attributes:

        :resname: The residue name.
        :resid: The residue id.
        :element: The element.
        :atomname: The atomname.

    reference_graph : networkx.Graph
        The reference graph as produced by ``make_reference``. Required node
        attributes:

        :resid: The residue id.
        :resname: The residue name.
        :found: The residue subgraph from the PDB file.
        :reference: The residue subgraph used as reference.
        :match: A dictionary describing how the reference corresponds
            with the provided graph. Keys are node indices of the
            reference, values are node indices of the provided graph.

    Returns
    -------
    networkx.Graph
        A new graph like ``aa_graph``, but with missing atoms (as per
        ``reference_graph``) added, and canonicalized atom and residue names.
    """
    mol = aa_graph.copy()
    for residx in reference_graph:
        # Rebuild missing atoms and canonicalize atomnames
        missing = []
        # Step 1: find all missing atoms. Canonicalize names while we're at it.
        reference = reference_graph.node[residx]['reference']
        match = reference_graph.node[residx]['match']
        chain = reference_graph.node[residx]['chain']
        resid = reference_graph.node[residx]['resid']
        resname = reference_graph.node[residx]['resname']
        for ref_idx in reference:
            if ref_idx in match:
                res_idx = match[ref_idx]
                node = mol.node[res_idx]
                node['atomname'] = reference.node[ref_idx]['atomname']
                node['element'] = reference.node[ref_idx]['element']
            else:
#                if reference.node[ref_idx]['element'] != 'H':
                print('Missing atom {}{}:{}'.format(resname, resid, reference.node[ref_idx]['atomname']))
                missing.append(ref_idx)
        # Step 2: try to add all missing atoms one by one. As long as we added
        # *something* the situation changed, and we might be able to place another.
        # We can only place atoms for which we know a neighbour.
        added = True
        while missing and added:
            added = False
            for ref_idx in missing:
                # See if the atom we want to add has a neighbour for which we know
                # the position. Otherwise, continue to the next.
                if all(ref_neighbour in missing for ref_neighbour in reference[ref_idx]):
                    continue
                added = True
                missing.pop(missing.index(ref_idx))
                res_idx = max(mol) + 1  # Alternative: find the first unused number

                # Create the new node
                match[ref_idx] = res_idx
                mol.add_node(res_idx)
                node = mol.node[res_idx]
                # TODO: Just copy all the attributes we have instead of listing
                # them everywhere. Maybe. We don't need all attributes (match,
                # found, reference).
#                node['position'] = np.zeros(3)
                node['chain'] = chain
                node['resname'] = resname
                node['resid'] = resid
                node['atomname'] = reference.node[ref_idx]['atomname']
                node['element'] = reference.node[ref_idx]['element']
#                print("Adding {}{}:{}".format(resname, resid, node['atomname']))

                neighbours = 0
                for neighbour_ref_idx in reference[ref_idx]:
                    try:
                        neighbour_res_idx = match[neighbour_ref_idx]
                    except KeyError:
                        continue
                    if not mol.has_edge(neighbour_res_idx, res_idx):
                        mol.add_edge(neighbour_res_idx, res_idx)
#                        node['position'] += mol.node[neighbour_res_idx]['position']
                        neighbours += 1
                assert neighbours != 0
#                if neighbours == 1:
#                    # Don't put atoms right on top of each other. Otherwise we'll
#                    # see some segfaults from MD software.
#                    node['position'] += np.random.normal(0, 0.01, size=3)
#                else:
#                    node['position'] /= neighbours
        if missing:
            for ref_idx in missing:
                print('Could not reconstruct atom {}{}:{}'.format(reference.node[ref_idx]['resname'],
                      reference.node[ref_idx]['resid'], reference.node[ref_idx]['atomname']))
    return mol


class RepairGraph(Processor):
    def run_molecule(self, molecule):
        reference_graph = make_reference(molecule)
        mol = repair_graph(molecule, reference_graph)
        return mol
