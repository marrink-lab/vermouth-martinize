#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:11:54 2017

@author: peterkroon
"""

from ..gmx import read_rtp
from ..molecule import Molecule
from .processor import Processor
from ..graph_utils import make_residue_graph

from collections import defaultdict
from itertools import product
import os

import networkx.algorithms.isomorphism as iso
import numpy as np


class GraphMapping:
    # Attributes to be removed from the blocks.
    forbidden = ['resid']

    # TODO: Add __getitem__, __iter__, __len__, __contains__, keys, values and
    #       items methods to emulate a Mapping?

    # TODO: Different methods of initializing the mapping? It might be nice to
    #       also provide the option to provide two molecules (instead) of lists
    #       of blocks and a mapping of {node_idx: [node_idx, ...], ...}

    # TODO: renumber output residues. Needs information about the entire system
    #       or we need to at least garantue we run it all in order. Which we
    #       can't unless we do run_system instead of run_molecule. We should
    #       maybe also move this class to a different file, but we'll see.
    def __init__(self, blocks_from, blocks_to, mapping):
        """
        blocks_from and blocks_to are sequences of Blocks.
        Mapping is a dictionary of {(residx, atomname): [(residx, atomname), ...], ...}.
        residx in these cases is the index of the residue in blocks_from and
        blocks_to respectively.
        """
        self.block_from = self._merge(blocks_from)
        self.block_to = self._merge(blocks_to)

        self.mapping = defaultdict(set)
        # Translate atomnames in mapping to node keys.
        for from_, to in mapping.items():
            res_from, name_from = from_
            for res_to, name_to in to:
                from_idxs = self.block_from.find_atoms(atomname=name_from, resid=res_from)
                to_idxs = self.block_to.find_atoms(atomname=name_to, resid=res_to)
                for to_idx in to_idxs:
                    self.mapping[to_idx].update(from_idxs)

        self.mapping = dict(self.mapping)

        # We can't do this in _merge, since we need the resids to translate the
        # mapping from (ambiguous) atomnames to (unique) graph keys. We do have
        # to get rid of them, otherwise they overwrite the resids of the graph
        # we're mapping (in do_mapping).
        self._purge_forbidden(self.block_from)
        self._purge_forbidden(self.block_to)

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
        out = blocks[0].to_molecule(resid=0)
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
            graph_mapping_collection[name] = GraphMapping(
                [from_ff.blocks[name], ],
                [to_ff.blocks[name], ],
                pair_mapping[name]
            )
    return graph_mapping_collection


def do_mapping(molecule, mappings, to_ff):
    pair_mapping = build_graph_mapping_collection(molecule.force_field, to_ff, mappings)

    residue_graph = make_residue_graph(molecule)
    graph_out = Molecule()
    bead_idx = 0
    residx_to_beads = defaultdict(set)

    for res_node_idx in residue_graph:
        residue = residue_graph.nodes[res_node_idx]
        graph = residue['graph']
        mapping = pair_mapping[residue['resname']]

        node_match = iso.categorical_node_match('atomname', '')

        # TODO: Reverse graph and block_from, and remove match inversion below.
        #       Can't do that right now, since we need to do
        #       subgraph_isomorphism, which is wrong (should be isomorphism).
        #       However, at time of writing we're still stuck with extraneous
        #       atoms such as termini, and no appropriate resnames and mappings.
        graphmatcher = iso.GraphMatcher(graph, mapping.block_from, node_match=node_match)

        matches = list(graphmatcher.subgraph_isomorphisms_iter())
        if len(matches) != 1:
            msg = ('Not one match ({}) for residue {}:{}.'
                   .format(len(matches), residue['resname'], res_node_idx))
            raise KeyError(msg)
        match = matches[0]

        match = {v: k for k, v in match.items()}  # TODO remove me. See above.

        mapped_match = {}
        for to_idx, from_idxs in mapping.mapping.items():
            mapped_match[to_idx] = [match[idx] for idx in from_idxs]
        # What we have now is a dict of {block_to_idx: [constructing_node_idxs]}

        block_to_bead_idx = {}
        for block_to_idx, from_idxs in mapped_match.items():
            # Needed for intra residue bonds
            block_to_bead_idx[block_to_idx] = bead_idx
            # Needed for inter residue bonds
            residx_to_beads[res_node_idx].add(bead_idx)
            bead = {}

            # Bead properties are taken from the last (!) atom, overwritten by
            # the block, and given a 'graph'
            # TODO: nx.quotient_graph?
            # FIXME: properties take from last bead instead of chosen 
            #        intellegently
            for n_idx in from_idxs:
                bead.update(graph.nodes[n_idx])
            bead.update(mapping.block_to.nodes[block_to_idx])
            bead['graph'] = graph.subgraph(from_idxs)
            assert bead_idx not in graph_out
            graph_out.add_node(bead_idx, **bead)

            bead_idx += 1
        # Make bonds within residue. We're not going to add interactions, we'll
        # leave that to the do_blocks processor since we might need to do
        # several mapping steps before we arive at the resolution we want.
        for block_idx, block_jdx in mapping.block_to.edges():
            bead_jdx = block_to_bead_idx[block_idx]
            bead_kdx = block_to_bead_idx[block_jdx]
            graph_out.add_edge(bead_jdx, bead_kdx)
    residx_to_beads = dict(residx_to_beads)

    # This makes edges between residues. We need to do this, since they can't
    # come from the mapping files and we need them to find the links locations.
    for res_idx, res_jdx in residue_graph.edges:
        for bd_idx, bd_jdx in product(residx_to_beads[res_idx],
                                      residx_to_beads[res_jdx]):
            if graph_out.has_edge(bd_idx, bd_jdx):
                # Continue, since their might be more bonds between these
                # residues
                continue
            for at_idx, at_jdx in product(graph_out.nodes[bd_idx]['graph'],
                                          graph_out.nodes[bd_jdx]['graph']):
                if molecule.has_edge(at_idx, at_jdx):
                    graph_out.add_edge(bd_idx, bd_jdx)
                    break  # On to the combination of beads

    graph_out._force_field = to_ff
    return graph_out


class DoMapping(Processor):
    def __init__(self, mappings, to_ff):
        self.mappings = mappings
        self.to_ff = to_ff
        super().__init__()

    def run_molecule(self, molecule):
        return do_mapping(molecule, mappings=self.mappings, to_ff=self.to_ff)

    def run_system(self, system):
        super().run_system(system)
        system.force_field = self.to_ff

