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

    def __init__(self, blocks_from, blocks_to, mapping):
        """
        blocks_from and blocks_to are sequences of Blocks.
        Mapping is a dictionary of {atomname: [atomnames]}. 
        """
        self.block_from = self._merge(blocks_from)
        self.block_to = self._merge(blocks_to)

        self.mapping = defaultdict(list)
        self._reverse_mapping = None
        
        # Translate atomnames in mapping to node keys.
        for name_from, names_to in mapping.items():
            for name_to in names_to:
                for to_idx in self.block_find_attr(self.block_to, name_to, 'atomname'):
                    self.mapping[to_idx].extend(self.block_find_attr(self.block_from, name_from, 'atomname'))
        self.mapping = dict(self.mapping)

        # Since we merged blocks, there may be edges missing in both ends. Add
        # from eachother.
        for to_idx, to_jdx in self.block_to.edges():
            self.block_from.add_edges_from(product(self.mapping[to_idx], self.mapping[to_jdx]))
#            for from_idx, from_jdx in product(self.mapping[to_idx], self.mapping[to_jdx]):
#                self.block_from.add_edges(from_idx, from_jdx)
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
        if self._reverse_mapping is None:
            self._make_reverse_map()
        return self._reverse_mapping

    def _make_reverse_map(self):
        if self._reverse_mapping is not None:
            return
        self._reverse_mapping = defaultdict(list)
        for to_idx, from_idxs in self.mapping.items():
            for from_idx in from_idxs:
                self._reverse_mapping[from_idx].append(to_idx)
        self._reverse_mapping = dict(self._reverse_mapping)

RTP_PATH = 'aminoacids.rtp'
with open(RTP_PATH) as rtp:
    blocks, links = read_rtp(rtp)

MAPPING = {}
for resname, block in blocks.items():
    mapping = {attrs['atomname']: [attrs['atomname'],] for attrs in block.atoms}
    MAPPING[resname] = GraphMapping([block], [block], mapping)

def get_mapping(resname):
    # TODO: Move this to files. Also, FIXME
    # CHARMM to CHARMM mapping to start of easy
    
    
#    BB = ('BB', set('C CA O N  HA HA1 HA2 H H1 H2 H3 OC1 OC2'.split()))
#    MAPPING = {
#           'GLY': [BB],
#           'LEU': [BB, ('SC1', set('CB CG CD1 CD2 1HB 2HB HG 1HD1 2HD1 3HD1 1HD2 2HD2 3HD2'.split()))],
#           'LYS': [BB, ('SC1', set('CB CG 1HB 2HB 1HG 2HG'.split())), ('SC2', set('CD CE NZ 1HD 2HD 1HE 2HE 1HZ 2HZ 3HZ'.split()))],
#           'PHE': [BB, ('SC1', set('CB CG 1HB 2HB '.split())), ('SC2', set('CD1 CE1 1HD 1HE'.split())), ('SC3', set('CZ CD2 CE2 2HE 2HD HZ'.split()))],
#           'CYS': [BB, ('SC1', set('CB SG'.split()))],
#           'HIS': [BB, ('SC1', set('CB CG'.split())), ('SC2', set('CD2 NE2'.split())), ('SC3', set('ND1 CE1'.split()))],
#           'SER': [BB, ('SC1', set('CB OG'.split()))],
#           'DPP': [('NC3', set('N C11 C12 C13 C14 C15'.split())),
#                   ('PO4', set('P O11 O12 O13 O14'.split())),
#                   ('GL1', set('C1 C2 O21 C21 O22'.split())),
#                   ('GL2', set('C3 O31 C31 O32'.split())),
#                   ('C1A', set('C22 C23 C24'.split())),
#                   ('C2A', set('C25 C26 C27 C28'.split())),
#                   ('C3A', set('C29 C210 C211 C212'.split())),
#                   ('C4A', set('C213 C214 C215 C216'.split())),
#                   ('C1B', set('C32 C33 C34'.split())),
#                   ('C2B', set('C35 C36 C37 C38'.split())),
#                   ('C3B', set('C39 C310 C311 C312'.split())),
#                   ('C4B', set('C313 C314 C315 C316'.split())),],
#           'DSB': [('BB', set('C O CA CB1 CB2'.split())),
#                   ('SC1', set('CZ SG1 CG1 CB1'.split())),
#                   ('SC2', set('CZ SG2 CG2 CB2'.split()))]
#           }
#    MAPPING['DTB'] = MAPPING['DSB']
    return MAPPING[resname]


def mean(graph, idxs, prop):
    return np.nanmean([graph.nodes[idx].get(prop, [np.nan]*3) for idx in idxs], axis=0)


def combine_with(graph, idxs, prop, func):
    return func([graph.nodes[idx][prop] for idx in idxs if prop in graph.nodes[idx]])


def do_mapping(molecule):
    residue_graph = make_residue_graph(molecule)
    graph_out = Molecule()
    bead_idx = 0
    residx_to_beads = defaultdict(set)

    for res_node_idx in residue_graph:
        residue = residue_graph.nodes[res_node_idx]
        graph = residue['graph']
        atname_to_idx = {graph.nodes[idx]['atomname']: idx for idx in graph}

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

            block_to_bead_idx[block_to_idx] = bead_idx
            residx_to_beads[res_node_idx].add(bead_idx)
            bead = {}

            for n_idx in from_idxs:
                bead.update(graph.nodes[n_idx])
            # Bead properties are take from the last (!) atom, overwritten by
            # the block, and given a 'graph'
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
        for bd_idx, bd_jdx in product(residx_to_beads[res_idx], residx_to_beads[res_jdx]):
            for at_idx, at_jdx in product(graph_out.nodes[bd_idx]['graph'], graph_out.nodes[bd_jdx]['graph']):
                if graph_out.has_edge(bd_idx, bd_jdx):
                    break
                if molecule.has_edge(at_idx, at_jdx):
                    graph_out.add_edge(bd_idx, bd_jdx)

    return graph_out


class DoMapping(Processor):
    def run_molecule(self, molecule):
        return do_mapping(molecule)
