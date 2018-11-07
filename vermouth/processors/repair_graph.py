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
Provides a processor that repairs a graph based on a reference.
"""
import networkx as nx

from .processor import Processor
from ..graph_utils import *
from ..ismags import ISMAGS
from ..log_helpers import StyleAdapter, get_logger
from ..utils import format_atom_string

LOGGER = StyleAdapter(get_logger(__name__))


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
        residue = residues.node[residx]['graph']
        reference = mol.force_field.reference_graphs[resname]
        add_element_attr(reference)
        add_element_attr(residue)
        # We are going to sort the nodes of reference and residue by atomname.
        # We do this, because the ISMAGS algorithm prefers to match nodes with
        # lower IDs. Note that short names sort before long names, so if there
        # are nodes missing it may still give the "wrong" answer.
        new_residue_names = {name: idx for idx, name in enumerate(sorted(residue, key=lambda jdx: residue.nodes[jdx].get('atomname', '')))}
        new_reference_names = {name: idx for idx, name in enumerate(sorted(reference, key=lambda jdx: reference.nodes[jdx].get('atomname', '')))}
        old_res_names = {v: k for k, v in new_residue_names.items()}
        old_ref_names = {v: k for k, v in new_reference_names.items()}

        res_copy = nx.relabel_nodes(residue, new_residue_names, copy=True)
        ref_copy = nx.relabel_nodes(reference, new_reference_names, copy=True)

        # Fun fact. If we assume res_copy > ref_copy, the tests run ~30 times
        # faster, and without failures. But, in real life examples, assuming
        # ref_copy > res_copy is a far better idea.
        ismags = ISMAGS(ref_copy, res_copy, node_match=nx.isomorphism.categorical_node_match('element', None))
        # Finding the largest common subgraph is expensive, but the first step
        # is to try and find a subgraph isomorphism between
        # residue >= reference, so best case it makes no difference, and worst
        # case we avoid trying to find that isomorphism twice.
        match_iter = ismags.largest_common_subgraph()
        matches = []
        best_score = None
        warned = False
        for match in match_iter:
            if not warned and len(match) != len(residue):
                LOGGER.debug('Doing MCS matching for residue {}{}', resname,
                             resid, type='performance')
                warned = True
            score = rate_match(ref_copy, res_copy, match)
            if best_score is None or score > best_score:
                best_score = score
                matches = [match]
            elif score == best_score:
                matches.append(match)
        for idx, match in enumerate(matches):
            # "unsort" the matches
            matches[idx] = {old_ref_names[ref]: old_res_names[res]
                            for ref, res in match.items()}

        if not matches:
            LOGGER.error("Can't find isomorphism between {}{} and its "
                         "reference.", resname, resid, type='inconsistent-data')
            raise ValueError
            continue

        matches = maxes(matches, key=lambda m: rate_match(reference, residue, m))
        if len(matches) > 1:
            LOGGER.warning("More than one way to fit {}{} on it's reference."
                           " I'm picking one arbitrarily. You might want to"
                           " fix at least some atomnames.", resname, resid,
                           type='bad-atom-names')

        match = matches[0]
        reference_graph.add_node(residx, chain=chain, reference=reference,
                                 found=residue, resname=resname, resid=resid,
                                 match=match)
    reference_graph.add_edges_from(residues.edges())
    return reference_graph


def repair_residue(molecule, ref_residue, include_graph):
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
            if include_graph:
                node['graph'] = molecule.subgraph([res_idx])
            node.update(reference.nodes[ref_idx])
            # Update found as well to keep found and molecule in line. It would
            # be better to try and figure why found is not a reference, but meh
            found.nodes[res_idx].update(reference.nodes[ref_idx])
        else:
            message = 'Missing atom {}{}:{}'
            args = (resname, resid, reference.nodes[ref_idx]['atomname'])
            if reference.nodes[ref_idx]['element'] != 'H':
                LOGGER.info(message, *args, type='missing-atom')
            else:
                # These are logged *below* debug level. Otherwise your screen
                # fills up pretty fast.
                LOGGER.log(5, message, *args, type='missing-atom')
            missing.append(ref_idx)
    # Step 2: try to add all missing atoms one by one. As long as we added
    # *something* the situation changed, and we might be able to place another.
    # We can only place atoms for which we know a neighbour.
    added = True
    while missing and added:
        added = False
        for ref_idx in missing:
            # See if the atom we want to add has a known neighbour. Otherwise,
            # continue to the next.
            if all(ref_neighbour in missing for ref_neighbour in reference[ref_idx]):
                continue
            added = True
            missing.pop(missing.index(ref_idx))
            # We don't find the lowest available number since that's just
            # asking for problems where you find an atom you don't expect
            # because the old one you were looking for was removed, and it's
            # number was reassigned.
            res_idx = max(molecule) + 1

            # Create the new node
            node = {}
            for key, val in ref_residue.items():
                # Some attributes are only relevant on a residue level, not on
                # an atom level.
                if key not in ('match', 'found', 'reference'):
                    node[key] = val
            node.update(reference.nodes[ref_idx])
            node['atomid'] = res_idx + 1

            match[ref_idx] = res_idx
            molecule.add_node(res_idx, **node)
            found.add_node(res_idx, **node)

            message = "Adding {}"
            args = format_atom_string(node)
            if node['element'] != 'H':
                LOGGER.debug(message, *args, type='missing-atom')
            else:
                # These are logged *below* debug level. Otherwise your screen
                # fills up pretty fast.
                LOGGER.log(5, message, args, type='missing-atom')

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
            LOGGER.error('Could not reconstruct atom {}{}:{}',
                         reference.nodes[ref_idx]['resname'],
                         reference.nodes[ref_idx]['resid'],
                         reference.nodes[ref_idx]['atomname'],
                         type='missing-atom')


def repair_graph(molecule, reference_graph, include_graph=True):
    """
    Repairs a molecule graph produced based on the information in
    ``reference_graph``. Missing atoms will be added and atom- and residue-
    names will be canonicalized. Atoms not present in ``reference_graph`` will
    have the attribute ``PTM_atom`` set to ``True``.

    `molecule` is modified in place. Missing atoms (as per `reference_graph`)
    are added, atom and residue names are canonicalized, and PTM atoms are
    marked.

    If `include_graph` is `True`, then the subgraph corresponding to each node
    is included in the node under the "graph" attribute.

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

    include_graph: bool
        Include the subgraph in the nodes.
    """
    for residx in reference_graph:
        residue = reference_graph.nodes[residx]
        repair_residue(molecule, residue, include_graph=include_graph)
        # Atomnames are canonized, and missing atoms added
        found = reference_graph.nodes[residx]['found']
        match = reference_graph.nodes[residx]['match']

        # Find the PTMs (or termini, or other additions) for *this* residue
        # `extra` is a set of the indices of the nodes from  `found` that have
        # no match in the reference graph.
        # `atachments` is a set of the nodes from `found` that have a match in
        # the reference and are connected to a node from `extra`.
        # We just stick a label on them for now, these are used by the PTM
        # processor.
        extra = set(found.nodes) - set(match.values())
        for idx in extra:
            molecule.nodes[idx]['PTM_atom'] = True
            found.nodes[idx]['PTM_atom'] = True

    return molecule


class RepairGraph(Processor):
    def __init__(self, delete_unknown=False, include_graph=True):
        super().__init__()
        self.delete_unknown = delete_unknown
        self.include_graph=include_graph

    def run_molecule(self, molecule):
        molecule = molecule.copy()
        reference_graph = make_reference(molecule)
        repair_graph(molecule, reference_graph, include_graph=self.include_graph)
        return molecule

    def run_system(self, system):
        mols = []
        for idx, molecule in enumerate(system.molecules):
            try:
                new_molecule = self.run_molecule(molecule)
            except KeyError as err:
                if not self.delete_unknown:
                    raise err
                else:
                    LOGGER.warning("Cannot recognize residue {} in  molecule {}. "
                                   "Deleting the molecule.",
                                   str(err), idx, type='unknown-residue')
            else:
                mols.append(new_molecule)
        system.molecules = mols
