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

import networkx.algorithms.isomorphism as iso
import numpy as np


class GraphMapping:
    forbidden = ['resid']

    # TODO: Add __getitem__, __iter__, __len__, __contains__, keys, values and
    #       items methods to emulate a Mapping?

    # FIXME
    def __init__(self, blocks_from, blocks_to, mapping):
        """
        blocks_from and blocks_to are sequences of Blocks.
        Mapping is a dictionary of {atomname: [atomnames]}. Since we can map
        n to m blocks, this is probably not good enough, and should also 
        contain an idx of which block it is.
        """
        self.block_from = self._merge(blocks_from)
        self.block_to = self._merge(blocks_to)

        self.mapping = defaultdict(set)
        # Translate atomnames in mapping to node keys.
        for name_from, names_to in mapping.items():
            for name_to in names_to:
                for to_idx in self.block_find_attr(self.block_to, name_to, 'atomname'):
                    from_idxs = self.block_find_attr(self.block_from, name_from, 'atomname')
                    self.mapping[to_idx].update(from_idxs)
        self.mapping = dict(self.mapping)
        self._make_reverse_map()

        # Since we merged blocks, there may be edges missing in both (between
        # the provided blocks). Add from eachother.
        for to_idx, to_jdx in self.block_to.edges():
            self.block_from.add_edges_from(product(self.mapping[to_idx],
                                                   self.mapping[to_jdx]))
        for from_idx, from_jdx in self.block_from.edges():
            self.block_to.add_edges_from(product(self.reverse_mapping[from_idx],
                                                 self.reverse_mapping[from_jdx]))

    @classmethod
    def _merge(cls, blocks):
        out = blocks[0].to_molecule()
        for block in blocks[1:]:
            out.merge_molecule(block)
        for n_idx in out:
            for attr in cls.forbidden:
                node = out.nodes[n_idx]
                if attr in node:
                    del node[attr]
        return out

    # TODO: Move to Block
    @staticmethod
    def block_find_attr(block, key, attr):
        for n_idx, attrs in block.nodes.items():
            if attrs[attr] == key:
                yield n_idx

    @property
    def reverse_mapping(self):
        if not hasattr(self, '_reverse_mapping'):
            self._make_reverse_map()
        return self._reverse_mapping

    def _make_reverse_map(self):
        self._reverse_mapping = defaultdict(set)
        for to_idx, from_idxs in self.mapping.items():
            for from_idx in from_idxs:
                self._reverse_mapping[from_idx].add(to_idx)
        self._reverse_mapping = dict(self._reverse_mapping)


def get_mapping(resname):
    return MAPPING[resname]


def do_mapping(molecule):
    residue_graph = make_residue_graph(molecule)
    graph_out = Molecule()
    bead_idx = 0
    residx_to_beads = defaultdict(set)

    for res_node_idx in residue_graph:
        residue = residue_graph.nodes[res_node_idx]
        graph = residue['graph']
        mapping = get_mapping(residue['resname'])

        node_match = iso.categorical_node_match('atomname', '')

        # TODO: Reverse graph and block_from, and remove match inversion below.
        #       Can't do that right now, since we need to do
        #       subgraph_isomorphism, which is wrong (should be isomorphism).
        #       However, at time of writing we're still stuck with extraneous
        #       atoms such as termini, and no appropriate resnames and mappings.
        graphmatcher = iso.GraphMatcher(graph, mapping.block_from, node_match=node_match)

        matches = list(graphmatcher.subgraph_isomorphisms_iter())
        assert len(matches) == 1
        match = matches[0]

        match = {v: k for k, v in match.items()}  # remove me. See above.

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

            # Bead properties are take from the last (!) atom, overwritten by
            # the block, and given a 'graph'
            for n_idx in from_idxs:
                bead.update(graph.nodes[n_idx])
            bead.update(mapping.block_to.nodes[block_to_idx])
            bead['graph'] = graph.subgraph(from_idxs)
            assert bead_idx not in graph_out
            graph_out.add_node(bead_idx, **bead)

            bead_idx += 1
        # Make bonds within residue. We're not going to add interactions, we'll
        # leave that to the do_blocks processor?
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
                    break  # On to the next bead

    return graph_out


RTP_PATH = 'aminoacids.rtp'
with open(RTP_PATH) as rtp:
    blocks, links = read_rtp(rtp)

MAPPING = {}
for resname, block in blocks.items():
    mapping = {attrs['atomname']: [attrs['atomname'],] for attrs in block.atoms}
    MAPPING[resname] = GraphMapping([block], [block], mapping)


class DoMapping(Processor):
    def run_molecule(self, molecule):
        return do_mapping(molecule)
