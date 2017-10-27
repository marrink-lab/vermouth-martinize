#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:39:20 2017

@author: peterkroon
"""
from ..gmx import read_rtp

from .processor import Processor
from ..graph_utils import make_residue_graph
from ..molecule import Molecule

from collections import ChainMap
from itertools import product


def apply_blocks(molecule, blocks):
    residue_graph = make_residue_graph(molecule)
    graph_out = Molecule()
    old_to_new_idxs = {}
    at_idx = 0
    for res_idx in residue_graph:
        residue = residue_graph.nodes[res_idx]
        res_graph = residue['graph']
        resname = residue['resname']
        block = blocks[resname]
        atname_to_idx = {}
        for atname in block:
            atom = list(res_graph.find_atoms(atname))

            assert len(atom) == 1
            old_to_new_idxs[atom[0]] = at_idx
            atname_to_idx[atname] = at_idx
            attrs = molecule.nodes[atom[0]]
            graph_out.add_node(at_idx, **ChainMap(block.nodes[atname], attrs))
            graph_out.nodes[at_idx]['graph'] = molecule.subgraph(atom)
            at_idx += 1
        for idx, jdx, data in block.edges(data=True):
            idx = atname_to_idx[idx]
            jdx = atname_to_idx[jdx]
            graph_out.add_edge(idx, jdx, **data)
        for inter_type, interactions in block.interactions.items():
            for interaction in interactions:
                atom_idxs = []
                for atom_name in interaction.atoms:
                    atom_idxs.extend(graph_out.find_atoms(atom_name,
                                                          resname=residue['resname'],
                                                          resid=residue['resid']))
                interactions = interaction._replace(atoms=atom_idxs)
                graph_out.add_interaction(inter_type, *interactions)

    # This makes edges between residues. We need to do this, since they can't
    # come from the blocks and we need them to find the links locations.
    for res_idx, res_jdx in residue_graph.edges():
        for old_idx, old_jdx in product(residue_graph.nodes[res_idx]['graph'],
                                        residue_graph.nodes[res_jdx]['graph']):
            try:
                # Usually termini, PTMs, etc
                idx = old_to_new_idxs[old_idx]
                jdx = old_to_new_idxs[old_jdx]
            except:
                continue
            if molecule.has_edge(old_idx, old_jdx):
                graph_out.add_edge(idx, jdx)
    return graph_out


RTP_PATH = '/usr/local/gromacs-2016.3/share/gromacs/top/charmm27.ff/aminoacids.rtp'


class ApplyBlocks(Processor):
    def run_molecule(self, molecule):
        with open(RTP_PATH) as rtp:
            blocks, links = read_rtp(rtp)
#        print(blocks)
        blocks['HIS'] = blocks['HSD']
        return apply_blocks(molecule, blocks)
