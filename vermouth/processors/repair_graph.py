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

from .processor import Processor
from ..graph_utils import *


import networkx as nx


def make_reference(mol):
    """
    Takes an molecule graph (e.g. as read from a PDB file), and finds and
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
            # INFO
            print('Doing MCS matching for residue {}{}'.format(resname, resid))
            # The problem is that some residues (termini in particular) will
            # contain more atoms than they should according to the reference.
            # Furthermore they will have too little atoms because X-Ray is
            # supposedly hard. This means we can't do the subgraph isomorphism
            # like we're used to. Instead, identify the atoms in the largest
            # common subgraph, and do the subgraph isomorphism/alignment on
            # those. MCS is ridiculously expensive, so we only do it when we
            # have to.
            try:
                mcs_match = max(maximum_common_subgraph(reference, residue, ['element']),
                                key=lambda m: rate_match(reference, residue, m))
            except ValueError:
                raise ValueError('No common subgraph found between {} and'
                                 'reference {}.'.format(resname, resname))
            # We could seed the isomorphism calculation with the knowledge from
            # the mcs_match, but thats to much effort for now.
            # TODO: see above
            res = residue.subgraph(mcs_match.values())
            matches = isomorphism(reference, res)
        # TODO: matches is sorted by isomorphism. So we should probably use
        #       that with e.g. itertools.takewhile.
        matches = maxes(matches, key=lambda m: rate_match(reference, residue, m))
        if not matches:
            raise ValueError("Can't find isomorphism between {}{} and it's "
                             "reference.".format(resname, resid))
        elif len(matches) > 1:
            # WARNING
            print("More than one way to fit {}{} on it's reference. I'm "
                  "picking one arbitrarily. You might want to fix at least "
                  "some atomnames.".format(resname, resid))
        match = matches[0]
        reference_graph.add_node(residx, chain=chain, reference=reference,
                                 found=residue, resname=resname, resid=resid,
                                 match=match)
    reference_graph.add_edges_from(residues.edges())
    return reference_graph


def repair_residue(molecule, ref_residue):
    """
    Rebuild missing atoms and canonicalize atomnames
    """
    # Rebuild missing atoms and canonicalize atomnames
    missing = []
    # Step 1: find all missing atoms. Canonicalize names while we're at it.
    reference = ref_residue['reference']
    found = ref_residue['found']
    match = ref_residue['match']

    resid = ref_residue['resid']
    resname = ref_residue['resname']

    for ref_idx in reference:
        if ref_idx in match:
            res_idx = match[ref_idx]
            node = molecule.nodes[res_idx]
            # Copy, because it's references everywhere.
            node['graph'] = molecule.subgraph([res_idx]).copy()
            node.update(reference.nodes[ref_idx])
            # Update found as well to keep found and molecule in line. It would
            # be better to try and figure why found is not a reference, but meh
            found.nodes[res_idx].update(reference.nodes[ref_idx])
        else:
            # if reference.nodes[ref_idx]['element'] != 'H':
            # INFO
            print(match)
            print('Missing atom {}{}:{}'.format(resname, resid, reference.nodes[ref_idx]['atomname']))
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
            res_idx = max(molecule) + 1  # Alternative: find the first unused number

            # Create the new node
            node = {}
            for key, val in ref_residue.items():
                # Some attributes are only relevant on a residue level, not on
                # an atom level.
                if key not in ('match', 'found', 'reference'):
                    node[key] = val
            node.update(reference.nodes[ref_idx])

            match[ref_idx] = res_idx
            molecule.add_node(res_idx, **node)
            found.add_node(res_idx, **node)
            # print("Adding {}{}:{}".format(resname, resid, node['atomname']))

            neighbours = 0
            for neighbour_ref_idx in reference[ref_idx]:
                try:
                    neighbour_res_idx = match[neighbour_ref_idx]
                except KeyError:
                    continue
                if not molecule.has_edge(neighbour_res_idx, res_idx):
                    molecule.add_edge(neighbour_res_idx, res_idx)
                    neighbours += 1
            assert neighbours != 0
    if missing:
        for ref_idx in missing:
            # WARNING?
            print('Could not reconstruct atom {}{}:{}'.format(reference.nodes[ref_idx]['resname'],
                  reference.nodes[ref_idx]['resid'], reference.nodes[ref_idx]['atomname']))


def repair_graph(molecule, reference_graph):
    """
    Repairs a molecule graph produced based on the
    information in ``reference_graph``. Missing atoms will be reconstructed and
    atom- and residue names will be canonicalized.

    Parameters
    ----------
    molecule : molecule.Molecule
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
        A new graph like ``molecule``, but with missing atoms (as per
        ``reference_graph``) added, and canonicalized atom and residue names.
    """
    for residx in reference_graph:
        residue = reference_graph.nodes[residx]
        repair_residue(molecule, residue)
        # Atomnames are canonized, and missing atoms added
        found = reference_graph.nodes[residx]['found']
        match = reference_graph.nodes[residx]['match']

        # Find the PTMs (or termini, or other additions) for *this* residue
        # `extra` is a set of the indices of the nodes from  `found` that have
        # no match in the reference graph.
        # `atachments` is a set of the nodes from `found` that have a match in
        # the reference and are connected to a node from `extra`.
        extra = set(found.nodes) - set(match.values())
        for idx in extra:
            molecule.nodes[idx]['PTM_atom'] = True
            found.nodes[idx]['PTM_atom'] = True

    return molecule


class RepairGraph(Processor):
    def __init__(self, delete_unknown=False):
        super().__init__()
        self.delete_unknown = delete_unknown

    def run_molecule(self, molecule):
        molecule = molecule.copy()
        reference_graph = make_reference(molecule)
        repair_graph(molecule, reference_graph)
        return molecule

    def run_system(self, system):
        mols = []
        for molecule in system.molecules:
            try:
                new_molecule = self.run_molecule(molecule)
            except KeyError as err:
                if not self.delete_unknown:
                    raise err
                else:
                    print("Can't recognize, deleting")
                    # TODO: raise a loud warning here
            else:
                mols.append(new_molecule)
        system.molecules = mols
