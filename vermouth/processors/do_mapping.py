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
Created on Tue Oct 10 11:11:54 2017

@author: peterkroon
"""

from ..gmx import read_rtp
from ..molecule import Molecule
from .processor import Processor
from ..graph_utils import make_residue_graph

from collections import defaultdict
from itertools import product, combinations

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np


class GraphMapping:
    # Attributes to be removed from the blocks.
    forbidden = ['resid', 'charge_group']

    # TODO: Add __getitem__, __iter__, __len__, __contains__, keys, values and
    #       items methods to emulate a Mapping?

    # TODO: Different methods of initializing the mapping? It might be nice to
    #       also provide the option to provide two molecules (instead) of lists
    #       of blocks and a mapping of {node_idx: [node_idx, ...], ...}

    # TODO: renumber output residues. Needs information about the entire system
    #       or we need to at least the garantee we run it all in order. Which we
    #       can't unless we do run_system instead of run_molecule. We should
    #       maybe also move this class to a different file, but we'll see.
    def __init__(self, blocks_from, blocks_to, mapping, weights=None, extra=()):
        """
        blocks_from and blocks_to are sequences of Blocks.
        Mapping is a dictionary of {(residx, atomname): [(residx, atomname), ...], ...}.
        residx in these cases is the index of the residue in blocks_from and
        blocks_to respectively.
        """
        if weights is None:
            weights = {}
        
        self.block_from = self._merge(blocks_from)
        self.block_to = self._merge(blocks_to)
#        for node in self.block_to:
#            self.block_to.nodes[node]['resid'] -= 1
#            self.block_to.nodes[node]['charge_group'] -= 1
        self.mapping = defaultdict(set)
        self.weights = defaultdict(dict)
        # Translate atomnames in mapping to node keys.
        for from_, to in mapping.items():
            res_from, name_from = from_
            from_idxs = list(self.block_from.find_atoms(atomname=name_from, resid=res_from+1))
            for res_to, name_to in to:
                to_idxs = self.block_to.find_atoms(atomname=name_to, resid=res_to+1)
                for to_idx in to_idxs:
                    self.mapping[to_idx].update(from_idxs)
                    if from_idxs:
                        self.weights[to_idx][from_idxs[0]] = (
                            weights[(res_to, name_to)][(res_from, name_from)]
                        )
        self._purge_forbidden(self.block_to)

        self.mapping = dict(self.mapping)
        self.extra = extra

        # We can't do this in _merge, since we need the resids to translate the
        # mapping from (ambiguous) atomnames to (unique) graph keys. We do have
        # to get rid of them, otherwise they overwrite the resids of the graph
        # we're mapping (in do_mapping).
        #self._purge_forbidden(self.block_from)
        #self._purge_forbidden(self.block_to)

        # Since we merged blocks, there may be edges missing in both (between
        # the provided blocks). Add from eachother.
        # Example: 
        #   from_blocks: a0-b0-c0 a1-b1-c1
        #   to_blocks:   A0-B0 C1-A1 B2-C2
        #   Mapping: {(0, A): [(0, a)], (0, B): [(0, b)], (1, C): [(0, c)],
        #             (1, A): [(1, a)], (2, B): [(1, b)], (2, C): [(1, c)]}
        # There are edges missing in both from_blocks and to_blocks (c0-A1, and
        # B0-C1 and A1-B2; respectively). There's enough information in the
        # "other" block to create those, so that's what we do.        
        #for to_idx, to_jdx in self.block_to.edges():
        #    self.block_from.add_edges_from(product(self.mapping[to_idx],
        #                                           self.mapping[to_jdx]))
            # e.g. for edge B0-C1 this is:
            # add_edges_from(product([(0, b)], [(0, c)])); which is the same as
            # add_edges_from(((0, b), (0, c)))
        # Cache the reverse map for a while. Maybe that means it shouldn't be
        # a property...
        #reverse_map = self.reverse_mapping
        # This loop does the same as the one above, but in the other direction.
        #for from_idx, from_jdx in self.block_from.edges():
        #    self.block_to.add_edges_from(product(reverse_map[from_idx],
        #                                         reverse_map[from_jdx]))

    @classmethod
    def _merge(cls, blocks):
        out = blocks[0].to_molecule()
        for block in blocks[1:]:
            out.merge_molecule(block)
        return out

    @classmethod
    def _purge_forbidden(cls, block):
        for n_idx in block:
            node = block.nodes[n_idx]
            for attr in cls.forbidden:
                if attr in node:
                    del node[attr]

    @property
    def reverse_mapping(self):
        reverse_mapping = defaultdict(set)
        for to_idx, from_idxs in self.mapping.items():
            for from_idx in from_idxs:
                reverse_mapping[from_idx].add(to_idx)
        reverse_mapping = dict(reverse_mapping)
        return reverse_mapping


def build_graph_mapping_collection(from_ff, to_ff, mappings):
    graph_mapping_collection = {}
    pair_mapping = mappings[from_ff.name][to_ff.name]
    for name in from_ff.blocks.keys():
        if name in to_ff.blocks and name in pair_mapping:
            mapping, weights, extra = pair_mapping[name]
            graph_mapping_collection[name] = GraphMapping(
                [from_ff.blocks[name], ],
                [to_ff.blocks[name], ],
                mapping,
                weights,
                extra,
            )
    return graph_mapping_collection


def are_all_equal(iterable):
    iterable = iter(iterable)
    first = next(iterable, None)
    return all(item == first for item in iterable)


def do_mapping(molecule, mappings, to_ff, attribute_keep=()):
    # We always keep the chain, the resid, and the resname from the original
    # molecule.
    graph_out = Molecule(forcefield=to_ff)
    attribute_keep = ['chain'] + list(attribute_keep)
    pair_mapping = build_graph_mapping_collection(molecule.force_field, to_ff, mappings)
    covered = defaultdict(int)
    print('===== START =====')
    all_matches = []
    for resname, mapping in pair_mapping.items():
        node_match = iso.categorical_node_match(['atomname', 'resname'], ['', ''])
        graphmatcher = iso.GraphMatcher(molecule, mapping.block_from, node_match=node_match)
        matches = list(graphmatcher.subgraph_isomorphisms_iter())
        for match in matches:
            for atom in match:
                covered[atom] += 1
            all_matches.append((match, resname, mapping))

    mol_to_out = defaultdict(list)
    for match, name, mapping in sorted(all_matches, key=lambda x: min(x[0].keys())):
        if graph_out.nrexcl is None:
            graph_out.nrexcl = mapping.block_to.nrexcl
        try:
            added_nodes = graph_out.merge_molecule(mapping.block_to)
        except ValueError as err:
            raise ValueError('Residue {} is not compatible with the'
                             ' others'.format(resname)) from err
        block_to_out = dict(zip(sorted(mapping.block_to.nodes), sorted(added_nodes)))
        block_to_mol = {v: k for k, v in match.items()}
        for to_idx, from_idxs in mapping.mapping.items():
            out_idx = block_to_out[to_idx]
            mol_idxs = [block_to_mol[from_idx] for from_idx in from_idxs]
            subgraph = molecule.subgraph(mol_idxs)
            graph_out.nodes[out_idx]['graph'] = subgraph
            attrs = {name: list(nx.get_node_attributes(subgraph, name).values())
                     for name in attribute_keep}
            print(attrs)
            for name, vals in attrs.items():
                if not are_all_equal(vals):
                    print('The attribute {} for atom {} is going to be garbage.'
                          ''.format(name, graph_out.nodes[out_idx]))
                if vals:
                    graph_out.nodes[out_idx][name] = vals[0]
                else:
                    graph_out.nodes[out_idx][name] = None
            for mol_idx in mol_idxs:
                mol_to_out[mol_idx].append(out_idx)

    mol_to_out = dict(mol_to_out)

    for match1, match2 in combinations(all_matches, 2):
        match1 = match1[0]
        match2 = match2[0]
        edges = molecule.edges_between(match1.keys(), match2.keys())
        if edges:
            for mol_idx, mol_jdx in edges:
                out_idxs = mol_to_out[mol_idx]
                out_jdxs = mol_to_out[mol_jdx]
                for out_idx, out_jdx in product(out_idxs, out_jdxs):
                    graph_out.add_edge(out_idx, out_jdx)
        shared_atoms = set(match1.keys()) & set(match2.keys())
        shared_out_atoms = [mol_to_out[mol_idx] for mol_idx in shared_atoms]
        for out_atoms in shared_out_atoms:
            if len(out_atoms) < 1:
                raise ValueError("This atom is shared between blocks, but only"
                                 " mapped once?")
            for out_idx, out_jdx in combinations(out_atoms, 2):
                graph_out.add_edge(out_idx, out_jdx)
        if shared_atoms:
            print("You have a shared atom between blocks. This may mean you"
                  " have too  particles in your output.")
    covered = dict(covered)
    print('double covered:', {k: v for k, v in covered.items() if v > 1})
    print('uncovered:', set(covered.keys()) - set(molecule.nodes))
    print(graph_out.nodes(data=True))
    print(graph_out.edges(data=True))
    print('=====  END  =====')
    return graph_out


class DoMapping(Processor):
    def __init__(self, mappings, to_ff, delete_unknown=False, attribute_keep=()):
        self.mappings = mappings
        self.to_ff = to_ff
        self.delete_unknown = delete_unknown
        self.attribute_keep = attribute_keep
        super().__init__()

    def run_molecule(self, molecule):
        return do_mapping(
            molecule,
            mappings=self.mappings,
            to_ff=self.to_ff,
            attribute_keep=self.attribute_keep
        )

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
                if new_molecule:
                    mols.append(new_molecule)
        system.molecules = mols
        system.force_field = self.to_ff

