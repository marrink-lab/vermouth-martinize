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
Created on Thu Jul 20 13:21:43 2017

@author: peterkroon
"""

from .gmx import *
from .pdb import *
from .utils import *
from .processors import *
from .system import *

import functools
import itertools
import os.path

import networkx as nx
import numpy as np


try:
    from mayavi import mlab
    from mayavi.modules.text3d import Text3D
    def draw(graph, ax=None, pos_key='position', name_key='atomname',
             node_color=(1, 0, 0), edge_color=None, node_size=300, width=1, 
             with_label=False, **kwargs):
        if edge_color is None:
            edge_color = node_color
        node_indices = sorted(graph)
        node_labels = [graph.node[n][name_key] for n in node_indices]
        poss = np.array([graph.node[n][pos_key] for n in node_indices])
        mlab.gcf().scene.disable_render = True
    
        edges = []
        for idx, jdx in graph.edges:
            edges.append((node_indices.index(idx), node_indices.index(jdx)))
    
        points = mlab.points3d(*poss.T, [node_size for _ in graph], color=node_color, scale_factor=0.025)
        points.mlab_source.dataset.lines = np.array(edges)
        points.mlab_source.update()
        tube = mlab.pipeline.tube(points, tube_radius=width/10)
        mlab.pipeline.surface(tube, color=edge_color)
        if with_label:
            for idx, label in enumerate(node_labels):
                mlab.text3d(*poss[idx], label)
    
        mlab.gcf().scene.disable_render = False
    show = mlab.show

except (RuntimeError, ImportError):
    import matplotlib.collections
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.close('all')
    fig = None
    ax = None

    def draw(graph, pos_key='position', name_key='atomname',
             node_color='r', edge_color=None, node_size=300, width=1,
             with_label=False, **kwargs):
        global fig, ax
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        if edge_color is None:
            edge_color = node_color
        positions = {n: graph.node[n][pos_key] for n in graph}
        node_labels = [graph.node[n][name_key] for n in graph]
        poss = np.array(list(positions.values()))
        ax.scatter(poss[:, 0], poss[:, 1], poss[:, 2], c=node_color, s=node_size, **kwargs)
        lines = []
        for idx, jdx in graph.edges:
            lines.append((positions[idx], positions[jdx]))
        if lines:
            lines = np.array(lines)
            line_coll = Line3DCollection(lines, color=edge_color, linewidth=width)
            ax.add_collection3d(line_coll)
        if with_label:
            for idx, label in enumerate(node_labels):
                ax.text(*poss[idx], label)
    show = plt.show

#pdb_filename = '../molecules/cycliclipopeptide_2.pdb'
#pdb_filename = '../molecules/6-macro-16.pdb'
#pdb_filename = '../molecules/6-macro-16.gro'
#pdb_filename = '../molecules/6-macro-8_cartwheel.gro'
#pdb_filename = '../molecules/6-macro-32.gro'
#pdb_filename = '../molecules/6-macro-32.pdb'
#pdb_filename = '../molecules/glkfk.pdb'

#IGNH = True

#Atom = namedtuple('Atom', ('atomid', 'atomname', 'altloc', 'resname', 'chain',
#                           'resid', 'insertion_code', 'coord', 'occupancy',
#                           'temp_factor', 'segmentid', 'element', 'charge'))

def modular_product(G, H):
    """
    Returns the `modular product`_  of graphs ``G`` and ``H``.
    
    Parameters
    ----------
    G: networkx.Graph
        A graph.
    H: networkx.Graph
        A graph.
    
    Returns
    -------
    networkx.Graph
        The modular product of ``G`` and ``H``.
    
    .. modular product:
        https://en.wikipedia.org/wiki/Modular_product_of_graphs
    """
    
    P = nx.tensor_product(G, H)
    for (u, v), (x, y) in itertools.combinations(P.nodes_iter(), 2):
        if u != x and v != y and not G.has_edge(u, x) and not H.has_edge(v, y):
            P.add_edge((u, v), (x, y))
    return P


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
    for (u1, v1), (u2, v2) in itertools.combinations(P.nodes_iter(), 2):
        both_edge = G.has_edge(u1, u2) and H.has_edge(v1, v2)
        neither_edge = not G.has_edge(u1, u2) and not H.has_edge(v1, v2)
        # Effectively: not (G.has_edge(u1, u2) xor H.has_edge(v1, v2))
        if u1 != u2 and v1 != v2 and (both_edge or neither_edge):
            attrs = {}
            if both_edge:
                for attr in set(G.edge[u1][u2].keys()) | set(H.edge[v1][v2].keys()):
                    attrs[attr] = (G.edge[u1][u2].get(attr, None), H.edge[v1][v2].get(attr, None))
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


def add_element_attr(molecule):
    for node_idx in molecule:
        node = molecule.node[node_idx]
        node['element'] = node.get('element', first_alpha(node['atomname']))


class NamedGraphMatcher(nx.isomorphism.GraphMatcher):
    def semantic_feasibility(self, node1, node2):
        # TODO: implement (partial) wildcards
        return self.G1.node[node1]['atomname'] == self.G2.node[node2]['atomname']


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


class LinkGraphMatcher(nx.isomorphism.GraphMatcher):
    def semantic_feasibility(self, node1, node2):
        bead = self.G1.node[node1]
        link = self.G2.node[node2]
        bead_atoms = set(bd['atomname'] for bd in bead['graph'].node.values())
        # TODO (partial) wildcards
        return bead['resname'] == link['resname'] and\
               bead['atomname'] == link['beadname'] and\
               any(atomname in bead_atoms for atomname in link['atomnames'])


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
        CG_mol.node[bead_idx]['position'] = np.mean([bd.node[idx]['position'] for idx in bd], axis=0)
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


def list_atoms(recursive_graph):
    for node_idx in recursive_graph.nodes_iter():
        node = recursive_graph.node[node_idx]
        if 'graph' in node:
            yield from list_atoms(node['graph'])
        else:
            yield node_idx


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
        # TODO: multiprocess this loop.
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
    reference_graph.add_edges_from(residues.edges_iter())
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
                if reference.node[ref_idx]['element'] != 'H':
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
                node['position'] = np.zeros(3)
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
                        node['position'] += mol.node[neighbour_res_idx]['position']
                        neighbours += 1
                assert neighbours != 0
                if neighbours == 1:
                    # Don't put atoms right on top of each other. Otherwise we'll
                    # see some segfaults from MD software.
                    node['position'] += np.random.normal(0, 0.01, size=3)
                else:
                    node['position'] /= neighbours
        if missing:
            for ref_idx in missing:
                print('Could not reconstruct atom {}{}:{}'.format(reference.node[ref_idx]['resname'],
                      reference.node[ref_idx]['resid'], reference.node[ref_idx]['atomname']))
    return mol

###############################################
# START THINGS THAT SHOULD BE READ FROM FILES #
###############################################

BB = ('BB', set('C CA O N  HA HA1 HA2 H H1 H2 H3 OC1 OC2'.split()))
MAPPING = {
           'GLY': [BB],
           'LEU': [BB, ('SC1', set('CB CG CD1 CD2 1HB 2HB HG 1HD1 2HD1 3HD1 1HD2 2HD2 3HD2'.split()))],
           'LYS': [BB, ('SC1', set('CB CG 1HB 2HB 1HG 2HG'.split())), ('SC2', set('CD CE NZ 1HD 2HD 1HE 2HE 1HZ 2HZ 3HZ'.split()))],
           'PHE': [BB, ('SC1', set('CB CG 1HB 2HB '.split())), ('SC2', set('CD1 CE1 1HD 1HE'.split())), ('SC3', set('CZ CD2 CE2 2HE 2HD HZ'.split()))]
           }

BB = ('BB', set('C CA O N'.split()))
MAPPING = {
           'GLY': [BB],
           'LEU': [BB, ('SC1', set('CB CG CD1 CD2'.split()))],
           'LYS': [BB, ('SC1', set('CB CG'.split())), ('SC2', set('CD CE NZ'.split()))],
           'PHE': [BB, ('SC1', set('CB CG'.split())), ('SC2', set('CD1 CE1'.split())), ('SC3', set('CZ CD2 CE2'.split()))]
           }

graph_mapping = {}
BB = nx.Graph(name='BB')
BB.add_node(0, atomname='N', element='N')
BB.add_node(1, atomname='CA', element='C')
BB.add_node(2, atomname='C', element='C')
BB.add_node(3, atomname='O', element='O')
BB.add_edges_from(((0, 1), (1, 2), (2,3)))
graph_mapping['GLY'] = [BB]
leu_sc1 = nx.Graph(name='SC1')
leu_sc1.add_nodes_from(range(4))
for idx, name in enumerate('CB CG CD1 CD2'.split()):
    leu_sc1.node[idx]['atomname'] = name
    leu_sc1.node[idx]['element'] = name[0]
leu_sc1.add_edges_from(((0, 1), (1, 2), (1, 3)))
graph_mapping['LEU'] = [BB, leu_sc1]
graph_mapping['LYS'] = [BB]

for bdname, atoms, edges in (('SC1', 'CB CG CD'.split(), ((0, 1), (1, 2))),
                             ('SC2', 'CD CE NZ'.split(), ((0, 1), (1, 2)))):
    bd = nx.Graph(name=bdname)
    for idx, atname in enumerate(atoms):
        bd.add_node(idx, atomname=atname, element=atname[0])
    bd.add_edges_from(edges)
    graph_mapping['LYS'].append(bd)

graph_mapping['PHE'] = [BB]

for bdname, atoms, edges in (['SC1', 'CB CG CD1 CD2'.split(), ((0, 1), (1, 2), (1, 3))],
                             ['SC2', 'CD1 CE1 CZ'.split(), ((0, 1), (1, 2))],
                             ['SC3', 'CZ CD2 CE2'.split(), ((0, 2), (1, 2))]):
    bd = nx.Graph(name=bdname)
    for idx, atname in enumerate(atoms):
        bd.add_node(idx, atomname=atname, element=atname[0])
    bd.add_edges_from(edges)
    graph_mapping['PHE'].append(bd)

graph_mapping['CYS'] = [BB]
for bdname, atoms, edges in (['SC1', 'CB SG'.split(), ((0, 1),)],):
    bd = nx.Graph(name=bdname)
    for idx, atname in enumerate(atoms):
        bd.add_node(idx, atomname=atname, element=first_alpha(atname))
    bd.add_edges_from(edges)
    graph_mapping['CYS'].append(bd)

graph_mapping['HIS'] = [BB]
for bdname, atoms, edges in (['SC1', 'CB CG'.split(), ((0, 1),)],
                             ['SC2', 'CD2 NE2'.split(), ((0, 1),)],
                             ['SC3', 'ND1 CE1'.split(), ((0, 1),)],
                             ):
    bd = nx.Graph(name=bdname)
    for idx, atname in enumerate(atoms):
        bd.add_node(idx, atomname=atname, element=first_alpha(atname))
    bd.add_edges_from(edges)
    graph_mapping['HIS'].append(bd)

graph_mapping['SER'] = [BB]
for bdname, atoms, edges in (['SC1', 'CB OG'.split(), ((0, 1),)],):
    bd = nx.Graph(name=bdname)
    for idx, atname in enumerate(atoms):
        bd.add_node(idx, atomname=atname, element=first_alpha(atname))
    bd.add_edges_from(edges)
    graph_mapping['SER'].append(bd)

graph_mapping['DPP'] = []
for bdname, atoms, edges in (['NC3', 'N C11 C12 C13 C14 C15'.split(), ((1, 2), (0, 2), (0, 3), (0, 4), (0, 5))],
                             ['PO4', 'P O11 O12 O13 O14'.split(), ((0, 1), (0, 2), (0, 3), (0, 4))],
                             ['GL1', 'C1 C2 O21 C21 O22'.split(), ((0, 1), (1, 2), (2, 3), (3, 4))],
                             ['GL2', 'C3 O31 C31 O32'.split(), ((0, 1), (1, 2), (2, 3))],
                             ['C1A', 'C22 C23 C24'.split(), ((0, 1), (1, 2))],
                             ['C2A', 'C25 C26 C27 C28'.split(), ((0, 1), (1, 2), (2, 3))],
                             ['C3A', 'C29 C210 C211 C212'.split(), ((0, 1), (1, 2), (2, 3))],
                             ['C4A', 'C213 C214 C215 C216'.split(), ((0, 1), (1, 2), (2, 3))],
                             ['C1B', 'C32 C33 C34'.split(), ((0, 1), (1, 2))],
                             ['C2B', 'C35 C36 C37 C38'.split(), ((0, 1), (1, 2), (2, 3))],
                             ['C3B', 'C39 C310 C311 C312'.split(), ((0, 1), (1, 2), (2, 3))],
                             ['C4B', 'C313 C314 C315 C316'.split(), ((0, 1), (1, 2), (2, 3))],
                             ):
    bd = nx.Graph(name=bdname)
    for idx, atname in enumerate(atoms):
        bd.add_node(idx, atomname=atname, element=first_alpha(atname))
    bd.add_edges_from(edges)
    graph_mapping['DPP'].append(bd)

graph_mapping['DSB'] = []
for bdname, atoms, edges in (['BB', 'C O CA CB1 CB2'.split(), ((0, 1), (0, 2), (2, 3), (2, 4))],
                             ['SC1', 'CZ SG1 CG1 CB1'.split(), ((0, 2), (1, 2), (2, 3))],
                             ['SC2', 'CZ SG2 CG2 CB2'.split(), ((0, 2), (1, 2), (2, 3))],
                             ):
    bd = nx.Graph(name=bdname)
    for idx, atname in enumerate(atoms):
        bd.add_node(idx, atomname=atname, element=first_alpha(atname))
    bd.add_edges_from(edges)
    graph_mapping['DSB'].append(bd)

graph_mapping['DTB'] = graph_mapping['DSB']

link = nx.Graph()
link.add_node(0, beadname='BB', resname='GLY', atomnames=set('CN'))
link.add_node(1, beadname='BB', resname='LEU', atomnames=set('CN'))
link.add_node(2, beadname='BB', resname='LYS', atomnames=set('CN'))
link.add_node(3, beadname='BB', resname='PHE', atomnames=set('CN'))
link.add_edges_from([(0, 1), (1, 2), (2, 3)])

#############################################
# END THINGS THAT SHOULD BE READ FROM FILES #
#############################################


def martinize(path, ignh=False):
    filename, ext = os.path.splitext(path)
    if ext == '.pdb':
        mol = read_pdb(path, exclude=('SOL', 'CL', 'NA'), ignh=ignh)
    elif ext == '.gro':
        mol = read_gro(path, exclude=('SOL', 'CL', 'NA'), ignh=ignh)
    else:
        print('Euh?')
    
    # For testing we can remove some atoms to see if we can rebuild them.
    #options = []
    #for idx in mol:
    #    # If you break the backbone connection you're f*cked.
    #    if mol.node[idx]['atomname'] not in 'C N'.split():
    #        options.append(idx)
    #
    ## STOP BLOWING HOLES IN MY GRAPH!!
    #remove = sorted(np.random.choice(options, size=0, replace=False))
    #print('Removing {}'.format(remove))
    #print("Removing "+'; '.join(["{}{}:{}".format(mol.node[n]['resname'], mol.node[n]['resid'], mol.node[n]['atomname']) for n in remove]))
    #draw(nx.subgraph(mol, remove), node_color=(1, 0.5, 0.5), node_size=30)
    #mol.remove_nodes_from(remove)

    reference_graph = make_reference(mol)
    mol = repair_graph(mol, reference_graph)
    
    residues = make_residue_graph(mol)
    
    
    draw(mol, node_size=30)
    draw(residues, node_size=100, node_color=(0, 1, 0), width=5)
    
    # Do some sanity checking. Sort the residue indices by chain id. We already
    # know that the residues graph should be sorted by resid
    # TODO: Check this!
    cur_chain = None
    residxs = []
    residxs_chain = []
    for res in residues:
        node = residues.node[res]
        if cur_chain is None:
            cur_chain = node['chain']
        if node['chain'] != cur_chain:
            residxs.append((cur_chain, residxs_chain))
            residxs_chain = []
            cur_chain = node['chain']
        residxs_chain.append(node['resid'])
    residxs.append((cur_chain, residxs_chain))

    # And now, for every chain, make sure the residue ids are consecutive
    for chain_id, idxs in residxs:
        cur_idx = idxs[0]
        for idx in idxs:
            if cur_idx != idx:
                print("Missing residue {} in chain {}".format(cur_idx, chain_id))
            else:
                cur_idx += 1
        # And while we're here, make sure every resid has only one residue type
        counts = np.bincount(idxs)
        wrong_idxs = np.where(counts > 1)[0]
        if wrong_idxs:
            print('Residue(s) {} in chain {} have multiple residue types.'.format(
                    ','.join(map(str, wrong_idxs)), chain_id))

    # Lastly, make sure every connected component has one chain ID, and vice
    # versa
    known_chain_ids = set()
    for connected_component in nx.connected_components(residues):
        chain_ids = set(residues.node[idx]['chain'] for idx in connected_component)
        if any(c_id in known_chain_ids for c_id in chain_ids) or len(chain_ids) > 1:
            print('Your chain IDs are messed up. Seek help.')
        known_chain_ids.update(chain_ids)

    # Start to do mapping
    # TODO: This needs serious refactoring. Part of it can be done by
    # leveraging the function blockmodel. Maybe. It should be a function anyway
    atidx2beadid = {}
    atidx2resid = {}
    CG_graph = nx.Graph()
    bead_idx = 0
    for residx in residues:
        resname = residues.node[residx]['resname']
        resid = residues.node[residx]['resid']
        try:
            mapping = graph_mapping[resname]
        except KeyError:
            print('Can\'t find mapping for residue {}'.format(resname))
            continue
        residue = residues.node[residx]['graph']
    
        # Build the CG nodes/beads
        for bead in mapping:
            name = bead.name
            GM = NamedGraphMatcher(residue, bead)
            matches = list(GM.subgraph_isomorphisms_iter())
            if len(matches) == 0:
                print('Relaxing criterion for {}{}:{}'.format(resname, resid, name))
                matches = isomorphism(residue, bead)
            match = matches[0]
            for idx, jdx in match.items():
                name1, name2 = residue.node[idx]['atomname'], bead.node[jdx]['atomname']
                if name1 != name2:
                    # Rectify found name
                    residue.node[idx]['atomname'] = name2
                    print('In {}{}:{}: matching {} to {}'.format(resname, resid, name, name1, name2))
            at_idxs = match.keys()
            # TODO: Make sure we find all the atoms we're looking for and vice
            # versa
    
            # Bookkeeping
            atidx2beadid.update(dict.fromkeys(at_idxs, bead_idx))
            atidx2resid.update(dict.fromkeys(at_idxs, residx))
    
            # Create CG node; analogous to blockmodel, but slightly different/more
            # flexible
            bd = residue.subgraph(at_idxs)
            CG_graph.add_node(bead_idx)
            CG_graph.node[bead_idx]['graph'] = bd
            # TODO: CoM instead of CoG
            CG_graph.node[bead_idx]['position'] = np.mean([bd.node[idx]['position'] for idx in bd], axis=0)
            CG_graph.node[bead_idx]['atomname'] = name
            CG_graph.node[bead_idx]['resid'] = resid
            CG_graph.node[bead_idx]['resname'] = resname
            CG_graph.node[bead_idx]['chain'] = bd.node[list(bd.node.keys())[0]]['chain']
            CG_graph.node[bead_idx]['nnodes'] = bd.number_of_nodes()
            CG_graph.node[bead_idx]['nedges'] = bd.number_of_edges()
            CG_graph.node[bead_idx]['density'] = nx.density(bd)
#            CG_graph.node[bead_idx]['charge'] = sum(float(bd.node[idx]['charge']) for idx in bd)
            bead_idx += 1
    
        # Build the edges within the CG residue
        # Should be read from the partial topology; and maybe give warning if
        # there's edges missing/superfluous?
        for idx, jdx, data in residue.edges_iter(data=True):
            try:
                cg_idx = atidx2beadid[idx]
                cg_jdx = atidx2beadid[jdx]
            except KeyError:
                # Atom not represented
                continue
            if cg_idx == cg_jdx:
                # No self loops; just to make sure
                continue
            weight = data.get('weight', 1.0)
            if CG_graph.has_edge(cg_idx, cg_jdx):
                CG_graph[cg_idx][cg_jdx]['weight'] += weight
            else:
                CG_graph.add_edge(cg_idx, cg_jdx, weight=weight)

    # Build the edges between residues. We need to do this to help the link
    # matching
    for residx, resjdx, data in residues.edges_iter(data=True):
        res1 = residues.node[residx]['graph']
        res2 = residues.node[resjdx]['graph']
    #    print(residues.node[residx]['resname'], residues.node[resjdx]['resname'])
        for at_idx, at_jdx in itertools.product(res1, res2):
            if mol.has_edge(at_idx, at_jdx):
                try:
                    cg_idx = atidx2beadid[at_idx]
                    cg_jdx = atidx2beadid[at_jdx]
                except KeyError:
                    print("You're going to have a problem! You're missing a bond between two residues.")
                    continue
                weight = mol[at_idx][at_jdx].get('weight', 1.0)
                if CG_graph.has_edge(cg_idx, cg_jdx):
                    CG_graph[cg_idx][cg_jdx]['weight'] += weight
                else:
                    CG_graph.add_edge(cg_idx, cg_jdx, weight=weight)
                n1 = CG_graph.node[cg_idx]
                n2 = CG_graph.node[cg_jdx]


    draw(CG_graph, node_size=70, node_color=(0, 0, 1), width=3)
    try:
        mlab.show()
    except:
        pass
    return CG_graph
#
#print('---')
#GM = LinkGraphMatcher(CG_graph, link)
#ms = list(GM.subgraph_isomorphisms_iter())
##print(ms)
#write_pdb(CG_graph, '{}_CG.pdb'.format(filename), conect=True)

