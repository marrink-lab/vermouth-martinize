#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:11:54 2017

@author: peterkroon
"""

from ..molecule import Molecule
from .processor import Processor
from ..graph_utils import make_residue_graph, blockmodel

from ..gmx import read_rtp

from collections import defaultdict
from functools import partial
from itertools import product
from operator import itemgetter

import numpy as np


def get_mapping(resname):
    BB = ('BB', set('C CA O N  HA HA1 HA2 H H1 H2 H3 OC1 OC2'.split()))
    MAPPING = {
           'GLY': [BB],
           'LEU': [BB, ('SC1', set('CB CG CD1 CD2 1HB 2HB HG 1HD1 2HD1 3HD1 1HD2 2HD2 3HD2'.split()))],
           'LYS': [BB, ('SC1', set('CB CG 1HB 2HB 1HG 2HG'.split())), ('SC2', set('CD CE NZ 1HD 2HD 1HE 2HE 1HZ 2HZ 3HZ'.split()))],
           'PHE': [BB, ('SC1', set('CB CG 1HB 2HB '.split())), ('SC2', set('CD1 CE1 1HD 1HE'.split())), ('SC3', set('CZ CD2 CE2 2HE 2HD HZ'.split()))],
           'CYS': [BB, ('SC1', set('CB SG'.split()))],
           'HIS': [BB, ('SC1', set('CB CG'.split())), ('SC2', set('CD2 NE2'.split())), ('SC3', set('ND1 CE1'.split()))],
           'SER': [BB, ('SC1', set('CB OG'.split()))],
           'DPP': [('NC3', set('N C11 C12 C13 C14 C15'.split())),
                   ('PO4', set('P O11 O12 O13 O14'.split())),
                   ('GL1', set('C1 C2 O21 C21 O22'.split())),
                   ('GL2', set('C3 O31 C31 O32'.split())),
                   ('C1A', set('C22 C23 C24'.split())),
                   ('C2A', set('C25 C26 C27 C28'.split())),
                   ('C3A', set('C29 C210 C211 C212'.split())),
                   ('C4A', set('C213 C214 C215 C216'.split())),
                   ('C1B', set('C32 C33 C34'.split())),
                   ('C2B', set('C35 C36 C37 C38'.split())),
                   ('C3B', set('C39 C310 C311 C312'.split())),
                   ('C4B', set('C313 C314 C315 C316'.split())),],
           'DSB': [('BB', set('C O CA CB1 CB2'.split())),
                   ('SC1', set('CZ SG1 CG1 CB1'.split())),
                   ('SC2', set('CZ SG2 CG2 CB2'.split()))]
           }
    MAPPING['DTB'] = MAPPING['DSB']
    return MAPPING[resname]


def mean(graph, idxs, prop):
    return np.nanmean([graph.nodes[idx].get(prop, [np.nan]*3) for idx in idxs], axis=0)


def combine_with(graph, idxs, prop, func):
    return func([graph.nodes[idx][prop] for idx in idxs if prop in graph.nodes[idx]])


def do_mapping(molecule, blocks):
    funcs = {'position': mean, 
             'charge': partial(combine_with, func=lambda l: sum(int(i[::-1]) if i else 0 for i in l)), 
             'chain': partial(combine_with, func=itemgetter(0)),
             'resid': partial(combine_with, func=itemgetter(0)),
             'resname': partial(combine_with, func=itemgetter(0))}
    residue_graph = make_residue_graph(molecule)
    graph_out = Molecule()
    bead_idx = 0
    residx_to_beads = defaultdict(set)
    for res_node_idx in residue_graph:
        residue = residue_graph.nodes[res_node_idx]
        graph = residue['graph']
        atoms = [graph.nodes[idx] for idx in graph]
        atname_to_idx = {graph.nodes[idx]['atomname']: idx for idx in graph}
        
        block = blocks[residue['resname']]
        mapping = get_mapping(residue['resname'])
        bdnames, atnames = list(zip(*mapping))
        
        idxs = []
        for atnames_per_bead in atnames:
            # TODO: handle missing (and extra?) atoms
            idxs.append([atname_to_idx[atname] for atname in atnames_per_bead if atname in atname_to_idx])
#        atidx_to_bdidx = {atidx: bead_idx+offset for offset, atidxs in enumerate(idxs) for atidx in atidxs}
        atidx_to_bdidx = {}
        for bdname, at_idxs in zip(bdnames, idxs):
            graph_out.add_node(bead_idx)
            bead = graph_out.nodes[bead_idx]
            atidx_to_bdidx.update({at_idx: bead_idx for at_idx in at_idxs})
            residx_to_beads[res_node_idx].add(bead_idx)

            # There must be a better way to do this.
            for prop in set(k for dk in map(dict.keys, atoms) for k in dk):
                if prop not in funcs:
                    continue
                func = funcs[prop]
                bead[prop] = func(graph, at_idxs, prop)

            bead['atomname'] = bdname
            bead['graph'] = graph.subgraph(at_idxs)
            bead_idx += 1
        
        # Add edges based on edges in the original graph
        for atidx, atjdx, data in graph.edges(data=True):
            if not (atidx in atidx_to_bdidx and atjdx in atidx_to_bdidx):
                continue
            bdidx = atidx_to_bdidx[atidx]
            bdjdx = atidx_to_bdidx[atjdx]
            if bdidx != bdjdx and not graph_out.has_edge(bdidx, bdjdx):
                graph_out.add_edge(bdidx, bdjdx)
        
        for inter_type, interactions in block.interactions:
            for interaction in interactions:
                atom_idxs = []
                for atom_name in interaction.atoms:
                    atom_idxs.extend(graph_out.find_atoms(atom_name,
                                                          resname=residue['resname'],
                                                          resid=residue['resid']))
                interactions = interaction._replace(atoms=atom_idxs)
                graph_out.add_interaction(inter_type, *interactions)
    # Add edges between residue based on edges in the original graph
    for res_idx, res_jdx in residue_graph.edges:
        for bd_idx, bd_jdx in product(residx_to_beads[res_idx], residx_to_beads[res_jdx]):
            for at_idx, at_jdx in product(graph_out.nodes[bd_idx]['graph'], graph_out.nodes[bd_jdx]['graph']):
                if graph_out.has_edge(bd_idx, bd_jdx):
                    break
                if molecule.has_edge(at_idx, at_jdx):
                    graph_out.add_edge(bd_idx, bd_jdx)
        
    return graph_out


RTP_PATH = '/usr/local/gromacs-2016.3/share/gromacs/top/charmm27.ff/aminoacids.rtp'


class DoMapping(Processor):
    def run_molecule(self, molecule):
        with open(RTP_PATH) as rtp:
            blocks, links = read_rtp(rtp)
        
        return do_mapping(molecule, blocks)
