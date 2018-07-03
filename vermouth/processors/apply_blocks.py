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
Provides a processor that adds interactions from blocks to molecules.
"""
# TODO: Move all this functionality to do_mapping?
from collections import ChainMap
from itertools import product

from .processor import Processor
from ..graph_utils import make_residue_graph
from ..molecule import Molecule


def apply_blocks(molecule, blocks):
    """
    Generate a new :class:`~vermouth.molecule.Molecule` based on the residue
    names and other attributes of `molecule` from `blocks`.

    Parameters
    ----------
    molecule: vermouth.molecule.Molecule
        The molecule to process.
    blocks: dict[str, vermouth.molecule.Block]
        The blocks known.

    Returns
    -------
    vermouth.molecule.Molecule
        A new molecule with attributes from the old `molecule`, as well as all
        interactions described by `blocks`.
    """
    graph_out = Molecule(
        force_field=molecule.force_field,
        meta=molecule.meta.copy()
    )
    residue_graph = make_residue_graph(molecule)

    # nrexcl may not be defined, but if it is we probably want to keep it
    try:
        graph_out.nrexcl = molecule.nrexcl
    except AttributeError:
        graph_out.nrexcl = None

    old_to_new_idxs = {}
    at_idx = 0
    charge_group_offset = 0
    for res_idx in residue_graph:
        residue = residue_graph.nodes[res_idx]
        res_graph = residue['graph']
        resname = residue['resname']
        block = blocks[resname]
        atname_to_idx = {}

        if graph_out.nrexcl is None:
            if hasattr(block, 'nrexcl'):
                graph_out.nrexcl = block.nrexcl
        else:
            if (hasattr(block, 'nrexcl')
                    and block.nrexcl is not None
                    and block.nrexcl != graph_out.nrexcl):
                raise ValueError('Not all blocks share the same value for "nrexcl".')

        for block_idx in block:
            atname = block.nodes[block_idx]['atomname']
            atom = list(res_graph.find_atoms(atomname=atname))
            assert len(atom) == 1, (block.name, atname, atom)
            old_to_new_idxs[atom[0]] = at_idx
            atname_to_idx[atname] = at_idx
            attrs = molecule.nodes[atom[0]]
            graph_out.add_node(at_idx, **ChainMap(block.nodes[atname], attrs))
            graph_out.nodes[at_idx]['graph'] = molecule.subgraph(atom)
            graph_out.nodes[at_idx]['charge_group'] += charge_group_offset
            graph_out.nodes[at_idx]['resid'] = attrs['resid']
            at_idx += 1
        charge_group_offset = graph_out.nodes[at_idx - 1]['charge_group']
        for idx, jdx, data in block.edges(data=True):
            idx = atname_to_idx[idx]
            jdx = atname_to_idx[jdx]
            graph_out.add_edge(idx, jdx, **data)
        for inter_type, interactions in block.interactions.items():
            for interaction in interactions:
                atom_idxs = []
                for atom_name in interaction.atoms:
                    atom_index = graph_out.find_atoms(atomname=atom_name,
                                                      resname=residue['resname'],
                                                      resid=residue['resid'])
                    atom_index = list(atom_index)
                    if not atom_index:
                        msg = ('Could not find a atom named "{}" '
                               'with resname being "{}" '
                               'and resid being "{}".')
                        raise ValueError(msg.format(atom_name, residue['resname'], residue['resid']))
                    atom_idxs.extend(atom_index)
                interactions = interaction._replace(atoms=atom_idxs)
                graph_out.add_interaction(inter_type, *interactions)

    # This makes edges between residues. We need to do this, since they can't
    # come from the blocks and we need them to find the links locations.
    # TODO This should not be done here, but by do_mapping, which might *also*
    #      do it at the moment
    for res_idx, res_jdx in residue_graph.edges():
        for old_idx, old_jdx in product(residue_graph.nodes[res_idx]['graph'],
                                        residue_graph.nodes[res_jdx]['graph']):
            try:
                # Usually termini, PTMs, etc
                idx = old_to_new_idxs[old_idx]
                jdx = old_to_new_idxs[old_jdx]
            except KeyError:
                continue
            if molecule.has_edge(old_idx, old_jdx):
                graph_out.add_edge(idx, jdx)
    return graph_out



class ApplyBlocks(Processor):
    def run_molecule(self, molecule):
        return apply_blocks(molecule, molecule.force_field.blocks)
