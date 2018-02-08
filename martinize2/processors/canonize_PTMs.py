#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:02:14 2018

@author: peterkroon
"""

from .processor import Processor

from collections import defaultdict
import itertools

import networkx as nx


# FIXME: read PTMs from files
KNOWN_PTMS = []
n_term = nx.Graph()  # Molecule? Block?
n_term.graph['name'] = 'N-terminus'
n_term.add_edges_from([(0, 1), (0, 2), (0, 3)])
n_term.nodes[0].update(atomname='N', element='N', PTM_atom=False)
n_term.nodes[1].update(atomname='HN', element='H', PTM_atom=False, rename='HN1')
n_term.nodes[2].update(atomname='HN2', element='H', PTM_atom=True)
n_term.nodes[3].update(atomname='HN3', element='H', PTM_atom=True)
KNOWN_PTMS.append(n_term)
c_term = nx.Graph()
c_term.graph['name'] = 'C-terminus'
c_term.add_edges_from([(0, 1), (0, 2)])
c_term.nodes[0].update(atomname='C', element='C', PTM_atom=False)
c_term.nodes[1].update(atomname='O', element='O', PTM_atom=False, rename='OC1')
c_term.nodes[2].update(atomname='OC2', element='O', PTM_atom=True)
KNOWN_PTMS.append(c_term)


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


def find_PTM_atoms(molecule):
    # Atomnames have already been fixed, and missing atoms have been added.
    # In addition, unrecognized atoms have been labeled with the PTM attribute.
    extra_atoms = set(n_idx for n_idx in molecule
                      if molecule.nodes[n_idx].get('PTM_atom', False))
    PTMs = []
    while extra_atoms:
        # First PTM atom we'll look at
        first = next(iter(extra_atoms))
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
            if orig in extra_atoms:
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
        extra_atoms -= atoms
        PTMs.append((atoms, attachments))
    return PTMs

# How to solve this graph covering problem
# Filter known_ptms, such that element_count(known_ptm) <= element_count(found)
# Filter known_ptms, such that known_ptm <= found (subgraph of). Keep the match
#   Not that this does mean that the PTMs in the PDB *must* be complete. So no
#   missing atoms.
# Find all the exactly covering combinations.
# Pick the best solution, such that the maximum size of the applied PTMs is
# maximal. (3, 2) > (3, 1, 1) > (2, 2, 1)  # Numbers are sizes of applied PTMs
#   If there are multiple best options, take the one with the most matching
#   atomnames


def identify_ptms(residue, residue_ptms, options):
    # BASECASE: residue_ptms is empty
    if not any(res_ptm[0] for res_ptm in residue_ptms):
        return []
    # REDUCTION: Apply one of options, remove those atoms from residue_ptms
    # COMBINATION: add the applied option to the output.
    for idx, option in enumerate(options):
        ptm, matcher = option
        matches = list(matcher.subgraph_isomorphisms_iter())
        # Matches: [{res_idxs: ptm_idxs}, {...}, ...]
        # All non-PTM atoms in residue are always available for matching...
        available = set(n_idx for n_idx in residue
                        if not residue.nodes[n_idx].get('PTM_atom', False))
        # ... and only add non consumed PTM atoms
        for res_ptm in residue_ptms:
            available.update(res_ptm[0])
        for match in matches:
            matching = set(match.keys())
            # FIXME: Currently we greedily take the first way it matches, even
            #        if this results in a dead-end later on. What we probably
            #        should do is expand `options` to all matches *before*
            #        calling this function. This should also change the output.
            #        On second thought, if every PTM has at least one anchor,
            #        this probably can't every become a problem.
            if matching.issubset(available):
                new_res_ptms = []
                for res_ptm in residue_ptms:
                    new_res_ptms.append((res_ptm[0] - matching, res_ptm[1]))
                # Continue with the remaining ptm atoms, and try just this
                # option and all smaller.
                return [(ptm, match)] + identify_ptms(residue, new_res_ptms, options[idx:])
    raise KeyError('Could not identify PTM')


def allowed_ptms(residue, res_ptms, known_ptms):
    # TODO: filter by element count first
    for ptm in known_ptms:
        ptm_graph_matcher = PTMGraphMatcher(residue, ptm)
        if ptm_graph_matcher.subgraph_is_isomorphic():
            yield ptm, ptm_graph_matcher


def fix_ptm(molecule):
    PTM_atoms = find_PTM_atoms(molecule)

    def key_func(ptm_atoms):
        node_idxs = ptm_atoms[-1]  # The anchors
        return sorted(molecule.nodes[idx]['resid'] for idx in node_idxs)

    ptm_atoms = sorted(PTM_atoms, key=key_func)

    resid_to_idxs = defaultdict(list)
    for n_idx in molecule:
        residx = molecule.nodes[n_idx]['resid']
        resid_to_idxs[residx].append(n_idx)
    resid_to_idxs = dict(resid_to_idxs)

    # FIXME: read from file
    known_PTMs = KNOWN_PTMS
    for resids, res_ptms in itertools.groupby(ptm_atoms, key_func):
        res_ptms = list(res_ptms)
        n_idxs = set()
        for resid in resids:
            n_idxs.update(resid_to_idxs[resid])
        # TODO: Maybe use graph_utils.make_residue_graph? Or rewrite that
        #       function?
        residue = molecule.subgraph(n_idxs)
        options = allowed_ptms(residue, res_ptms, known_PTMs)
        # TODO/FIXME: This includes anchors in sorting by size.
        options = sorted(options, key=lambda opt: len(opt[0]))
        identified = identify_ptms(residue, res_ptms, options)
        print('Today your answer is: {}!'.format([out[0].graph['name'] for out in identified]))
        for ptm, match in identified:
            for mol_idx, ptm_idx in match.items():
                ptm_node = ptm.nodes[ptm_idx]
                mol_node = molecule.nodes[mol_idx]
                if ptm_node['PTM_atom'] or 'rename' in ptm_node:
                    mol_node['graph'] = molecule.subgraph([mol_idx]).copy()
                    new_name = ptm_node.get('rename', ptm_node['atomname'])
                    print('Renaming {} to {}'.format(mol_node['atomname'], new_name))
                    mol_node['atomname'] = new_name
            for n_idx in n_idxs:
                molecule.nodes[n_idx]['modifications'] = molecule.nodes[n_idx].get('modifications', [])
                molecule.nodes[n_idx]['modifications'].append(ptm)
    return molecule


class CanonizePTMs(Processor):
    def run_molecule(self, molecule):
        print(len(molecule))
        molecule = fix_ptm(molecule)
        return molecule
