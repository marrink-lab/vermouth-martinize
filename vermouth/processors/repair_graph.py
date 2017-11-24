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

from collections import defaultdict
import functools
import os.path

import networkx as nx


class PTMGraphMatcher(nx.isomorphism.GraphMatcher):
    # G1 >= G2; G1 is the found residue; G2 the PTM reference
    def semantic_feasibility(self, node1, node2):
        node1 = self.G1.nodes[node1]
        node2 = self.G2.nodes[node2]
        if node1.get('PTM_atom', False) == node2['PTM_atom']:
            if node2['PTM_atom']:
                # elements must match
                return node1['element'] == node2['element']
            else:
                # atomnames must match
                return node1['atomname'] == node2['atomname']
        else:
            return False


def update_reference(reference, found, residue_ptms):
    global KNOWN_PTMS
    # residue_ptms = ((atom_idxs, attachment_idxs), ...)
    extra_atom_idxs = set()
    for atom_idxs, attachment_idxs in residue_ptms:
        extra_atom_idxs |= atom_idxs
    idx = next(iter(found))
    resname = found.nodes[idx]['resname']
    found_to_ref = {v: k for k, v in reference['match'].items()}
    # resid = found.nodes[idx]['resid']

    # TODO: Relabel all nodes in reference['reference'] (and reference['match'])
    #       to integer? Otherwise it's going to be a little tricky to ensure
    #       all atom names will remain unique between all combinations of
    #       residues and PTMs.

    # We have here a graph covering problem: How can we (best) cover all ptm
    # atoms found with known PTMs?
    # For now we solve it by greedily trying the largest known PTMs and
    # removing atoms recognized. That way there can't be overlapping PTMs. If
    # there's still PTM atoms left at the end of the ride we raise an error.
    KNOWN_PTMS = sorted(KNOWN_PTMS, key=len)
    available = found.copy()
    while extra_atom_idxs:
        found_ptm = False
        for PTM_template in KNOWN_PTMS:
            # PTMs _must_ be smaller or equal to their references. A PTM can't have
            # another PTM.
            # TODO: filter possible PTMs based on e.g. element count. This does
            #       require a few brain-cycles.
            # FIXME: This graphmatch will FAIL if there's PTM atoms missing.
            #        Instead we should (always?) do a MCS step, and properly
            #        seed that with the anchor(s), and only do the combination
            #        between the PTM atoms.
            gm = PTMGraphMatcher(available, PTM_template)
            matches = gm.subgraph_isomorphisms_iter()
            matches = maxes(matches, key=lambda m: rate_match(found, PTM_template, m))
            if len(matches) > 1:
                # WARNING?
                print('More than one way to apply my PTM. Fix some of your atom '
                      'names if I guess wrong.')
            elif not matches:
                # This PTM did not match.
                continue
            found_ptm = True
            match = matches[0]
            ptm_ref_to_found = {v: k for k, v in match.items()}
            available.remove_nodes_from(ptm_ref_to_found.values())
            extra_atom_idxs -= set(ptm_ref_to_found.values())

            # Start updating the reference
            ptm_to_reference = {}  # Needed for edges.
            for ptm_idx in PTM_template:
                ptm_node = PTM_template.nodes[ptm_idx].copy()
                if 'rename' in ptm_node:
                    ptm_node['atomname'] = ptm_node.pop('rename')

                if ptm_idx not in ptm_ref_to_found:
                    # PTM atom not found. Add it to the reference, and let
                    # repair_residue rebuild it.
                    # There's 2 cases here: it's already in reference, in which
                    # case it has already been rebuild, or it's a PTM atom.
                    # Since non-PTM atoms have already been repaired those will
                    # never be missing.
                    assert ptm_node['atomname'] not in reference['reference']
                    reference['reference'].add_node(ptm_node['atomname'], **ptm_node)
                    ptm_to_reference[ptm_idx] = ptm_node['atomname']
                    continue

                # PTM atom found
                mol_idx = ptm_ref_to_found[ptm_idx]
                if not ptm_node['PTM_atom']:
                    # Atom should be in the reference. Might be renamed though.
                    ref_idx = found_to_ref[mol_idx]
                    ref_node = reference['reference'].nodes[ref_idx]
                    ref_node.update(**ptm_node)
                    print('Renaming reference node {} to {}'.format(ref_idx, ptm_node['atomname']))
                    assert ptm_node['atomname'] not in reference['reference'] or ref_idx == ptm_node['atomname']
                    # Relabel reference node
                    nx.relabel_nodes(reference['reference'], {ref_idx: ptm_node['atomname']}, copy=False)
                    # And rename reference['match']
                    reference['match'][ptm_node['atomname']] = reference['match'].pop(ref_idx)
                    # ref_idx is already renamed to ptm_node['atomname']
                    ptm_to_reference[ptm_idx] = ptm_node['atomname']
                else:
                    # Atom is not known to reference, so add a node to
                    # reference.
                    print('Adding reference node {}'.format(ptm_node['atomname']))
                    assert ptm_node['atomname'] not in reference['reference']
                    reference['reference'].add_node(ptm_node['atomname'], **ptm_node)
                    reference['match'][ptm_node['atomname']] = ptm_ref_to_found[ptm_idx]
                    ptm_to_reference[ptm_idx] = ptm_node['atomname']
            # All PTM nodes should now be present in the reference. Time to add
            # the edges.
            for ptm_idx, ptm_jdx, data in PTM_template.edges(data=True):
                ref_idx = ptm_to_reference[ptm_idx]
                ref_jdx = ptm_to_reference[ptm_jdx]
                print('Adding edge between {} and {}'.format(ref_idx, ref_jdx))
                reference['reference'].add_edge(ref_idx, ref_jdx, **data)
        if not found_ptm:
            break
    if extra_atom_idxs:
        raise ValueError('Could not cover all PTMs found.')

    # We modified reference, so return None.
    return None


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
            # supposedly hard. This means we can't do the subgraph isomorphism like
            # we're used to. Instead, identify the atoms in the largest common
            # subgraph, and do the subgraph isomorphism/alignment on those. MCS is
            # ridiculously expensive, so we only do it when we have to.
            try:
                mcs_match = max(maximum_common_subgraph(reference, residue, ['element']),
                                key=lambda m: rate_match(reference, residue, m))
            except ValueError:
                raise ValueError('No common subgraph found between {} and'
                                 'reference {}.'.format(resname, resname))
            # We could seed the isomorphism calculation with the knowledge from the
            # mcs_match, but thats to much effort for now.
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
#            if reference.nodes[ref_idx]['element'] != 'H':
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
    PTMs = []
    for residx in reference_graph:
        residue = reference_graph.nodes[residx]
        repair_residue(molecule, residue)
        # Atomnames are canonized, and missing atoms added
        found = reference_graph.nodes[residx]['found']
        match = reference_graph.nodes[residx]['match']
        resid = reference_graph.nodes[residx]['resid']
        
        # Find the PTMs (or termini, or other additions) for *this* residue
        # `extra` is a set of the indices of the nodes from  `found` that have
        # no match in the reference graph.
        # `atachments` is a set of the nodes from `found` that have a match in
        # the reference and are connected to a node from `extra`.
        extra = set(found.nodes) - set(match.values())
        for idx in extra:
            molecule.nodes[idx]['PTM_atom'] = True
            found.nodes[idx]['PTM_atom'] = True
        residue_ptms = []
        while extra:
            # First PTM atom we'll look at
            first = next(iter(extra))
            attachments = set()
            # PTM atoms we've found
            atoms = set()
            # Atoms we still need to see this traversal
            to_see = set([first])
            # Traverse in molecule. Since extra is limited to this residue
            # we'll at most see anchors in other residues.
            for orig, succ in nx.bfs_successors(molecule, first):
                # We've seen orig, so remove it
                to_see.remove(orig)
                if orig in extra:
                    # If this is a PTM atom, we want to see it's neighbours as
                    # well.
                    to_see.update(succ)
                    atoms.add(orig)
                else:
                    # Else, it's an attachment point for the this PTM
                    attachments.add(orig)
                if not to_see:
                    # We've traversed the interesting bit of the tree
                    break
            extra -= atoms
            # TODO: If the attachments/anchors are in different residues, we
            #       should treat them as one.
            residue_ptms.append((atoms, attachments))
        if residue_ptms:
            PTMs.append((found, tuple(residue_ptms)))
    # All residues have been canonicalized. Now we can go and find our PTMs.
    # What we should do is find which PTM this is, get/make a new reference for
    # the affected residues, and call repair_residue on them again.
    # For now, just print and remove them...

    # How we're going to do this:
    # 1) Find correct PTM, which should be a (small) Graph where atoms are
    #    marked as either part of the "original" residue, or the PTM
    # 2) Do a graph isomorphism (note: atoms might be missing?), making sure
    #    atoms marked as part of the PTM are in `extra` and vice versa.
    # 3) Canonicalize the atoms of the PTM.
    # 4) We need to keep track of which PTM lives where. Some PTMs might make
    #    separate residues (open problem). Either way, give *every* atom of the
    #    residue a 'PTM' attribute identifying the PTM. This
    #    can/will/should/needs to be used in the later processors (e.g. mapping
    #    and blocks).

    for found, residue_ptms in PTMs:
        
        idx = next(iter(found))  # Pick an arbitrary atom
        resname = found.nodes[idx]['resname']
        resid = found.nodes[idx]['resid']
        ref_idx = [idx for idx in reference_graph if reference_graph.nodes[idx]['resid'] == resid][0]
        reference = reference_graph.nodes[ref_idx]
        # Because multiple ptms could actually be one we group them per residue
        # E.g. protonated N terminus
        update_reference(reference, found, residue_ptms)
        repair_residue(molecule, reference)
#        repair_ptm(molecule, found, residue_ptms)
        # INFO
#        print('Extra atoms for residue {}{}:'.format(resname, resid))
#        for atom_idxs, attachment_idxs in residue_ptms:
#
#            print('\t', [molecule.nodes[idx]['atomname'] for idx in atom_idxs], end='; ')
#            for n_idx in attachment_idxs:
#                resname = molecule.nodes[n_idx]['resname']
#                resid = molecule.nodes[n_idx]['resid']
#                atomname = molecule.nodes[n_idx]['atomname']
#                print('Attached to: {}{}:{}'.format(resname, resid, atomname), end=', ')
#            print()
#            # WARNING
#            print("Couldn't recognize this PTM, removing atoms involved.")
#            molecule.remove_nodes_from(atom_idxs)


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
                    # TODO: raise a loud warning here
                    pass
            else:
                mols.append(new_molecule)
        system.molecules = mols
