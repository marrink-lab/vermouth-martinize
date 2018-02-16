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
Created on Thu Oct  5 10:43:19 2017

@author: peterkroon
"""

from ..molecule import Molecule
from .processor import Processor
from ..utils import first_alpha, maxes
from ..graph_utils import *
from ..gmx import read_rtp

import functools
import itertools
import os.path

import networkx as nx


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
        # check whether the node degrees match?

        resname = residues.node[residx]['resname']
        resid = residues.node[residx]['resid']
        chain = residues.node[residx]['chain']
        # print("{}{}".format(resname, resid), flush=True)
        residue = residues.node[residx]['graph']
        reference = mol.force_field.reference_graphs[resname]
        add_element_attr(reference)
        add_element_attr(residue)
        # Assume reference >= residue
        matches = isomorphism(reference, residue)
        if not matches:
            # Maybe reference < residue? I.e. PTM or protonation
            matches = isomorphism(residue, reference)
            matches = [{v: k for k, v in match.items()} for match in matches]
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
                mcs_match = max(maximum_common_subgraph(reference, residue, ['element']),
                                key=lambda m: rate_match(reference, residue, m))
            except ValueError:
                raise ValueError('No common subgraph found between {} and reference {}'.format(resname, resname))
#            print('Did MCS matching for residue {}{}'.format(resname, resid))
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
                node['graph'] = aa_graph.subgraph([res_idx])
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
    def __init__(self, delete_unknown=False):
        super().__init__()
        self.delete_unknown = delete_unknown

    def run_molecule(self, molecule):
        reference_graph = make_reference(molecule)
        mol = repair_graph(molecule, reference_graph)
        return mol

    def run_system(self, system):
        mols = []
        for molecule in system.molecules:
            try:
                new_molecule = self.run_molecule(molecule)
            except KeyError as err:
                if not self.delete_unknown:
                    raise err
                else:
                    # TODO: raise a loud warning here
                    pass
            else:
                mols.append(new_molecule)
        system.molecules = mols
