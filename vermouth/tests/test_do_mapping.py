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

from collections import defaultdict

from vermouth.processors.do_mapping import do_mapping
import vermouth.forcefield
from vermouth.molecule import Molecule, Block
import networkx.algorithms.isomorphism as iso


FF_MARTINI = vermouth.forcefield.get_native_force_field(name='martini22')
FF_UNIVERSAL = vermouth.forcefield.get_native_force_field(name='universal')

AA_MOL = Molecule(force_field=FF_UNIVERSAL)
AA_MOL.add_nodes_from((
        (0, {'resid': 1, 'resname': 'IPO', 'atomname': 'C1', 'chain': 'A', 'element': 'C'}),
        (1, {'resid': 1, 'resname': 'IPO', 'atomname': 'C2', 'chain': 'A', 'element': 'C'}),
        (2, {'resid': 1, 'resname': 'IPO', 'atomname': 'C3', 'chain': 'A', 'element': 'C'}),
        (3, {'resid': 2, 'resname': 'IPO', 'atomname': 'C1', 'chain': 'A', 'element': 'C'}),
        (4, {'resid': 2, 'resname': 'IPO', 'atomname': 'C2', 'chain': 'A', 'element': 'C'}),
        (5, {'resid': 2, 'resname': 'IPO', 'atomname': 'C3', 'chain': 'A', 'element': 'C'}),
        (6, {'resid': 3, 'resname': 'IPO', 'atomname': 'C1', 'chain': 'A', 'element': 'C'}),
        (7, {'resid': 3, 'resname': 'IPO', 'atomname': 'C2', 'chain': 'A', 'element': 'C'}),
        (8, {'resid': 3, 'resname': 'IPO', 'atomname': 'C3', 'chain': 'A', 'element': 'C'}),
        ))
AA_MOL.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6), (6, 7), (7, 8)])
block_aa = Block(force_field=FF_UNIVERSAL)
block_aa.name = 'IPO'
block_aa.add_nodes_from((
    ('C1', {'resid': 1, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C2', {'resid': 1, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C3', {'resid': 1, 'resname': 'IPO', 'atomname': 'C3'}),))
block_aa.add_edges_from([('C1', 'C2'), ('C2', 'C3')])

block_cg = Block(force_field=FF_MARTINI)
block_cg.name = 'IPO'
block_cg.add_nodes_from((('B1', {'resid': 1, 'resname': 'IPO', 'atomname': 'B1'}),))

FF_MARTINI.blocks['IPO'] = block_cg
FF_UNIVERSAL.blocks['IPO'] = block_aa


block_aa = Block(force_field=FF_UNIVERSAL)
block_aa.name = 'IPO_large'
block_aa.add_nodes_from((
    ('C1', {'resid': 1, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C2', {'resid': 1, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C3', {'resid': 1, 'resname': 'IPO', 'atomname': 'C3'}),
    ('C4', {'resid': 2, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C5', {'resid': 2, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C6', {'resid': 2, 'resname': 'IPO', 'atomname': 'C3'}),
    ('C7', {'resid': 3, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C8', {'resid': 3, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C9', {'resid': 3, 'resname': 'IPO', 'atomname': 'C3'}),))
block_aa.add_edges_from([('C1', 'C2'), ('C2', 'C3'), ('C2', 'C4'),
                         ('C4', 'C5'), ('C5', 'C6'), ('C5', 'C7'),
                         ('C7', 'C8'), ('C8', 'C9'),
                         ])

block_cg = Block(force_field=FF_MARTINI)
block_cg.name = 'IPO_large'
block_cg.add_nodes_from((('B1', {'resid': 1, 'resname': 'IPO_large', 'atomname': 'B1'}),
                         ('B2', {'resid': 2, 'resname': 'IPO_large', 'atomname': 'B1'}),
                         ('B3', {'resid': 3, 'resname': 'IPO_large', 'atomname': 'B1'}),))
block_cg.add_edges_from([('B1', 'B2'), ('B2', 'B3')])

FF_MARTINI.blocks['IPO_large'] = block_cg
FF_UNIVERSAL.blocks['IPO_large'] = block_aa


def _equal_graphs(g1, g2):
    attrs = ['resid', 'resname', 'atomname', 'chain', 'charge_group', 'atype']
    node_equal = iso.categorical_node_match(attrs, ['']*len(attrs))
    matcher = iso.GraphMatcher(g1, g2, node_match=node_equal)
    return matcher.is_isomorphic()


def test_no_residue_crossing():
    """
    Make sure we don't cross residue boundaries
    """
    mapping = {(0, 'C1'): [(0, 'B1')], (0, 'C2'): [(0, 'B1')], (0, 'C3'): [(0, 'B1')]}
    weights = {(0, 'B1'): {(0, 'C1'): 1, (0, 'C2'): 1, (0, 'C3'): 1, }}
    extra = ()
    mappings = {'universal': {'martini22': {'IPO': (mapping, weights, extra)}}}

    cg = do_mapping(AA_MOL, mappings, FF_MARTINI)

    expected = Molecule(force_field=FF_MARTINI)
    expected.add_nodes_from((
        (0, {'resid': 1, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
        (1, {'resid': 2, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 2}),
        (2, {'resid': 3, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 3}),
            ))
    expected.add_edges_from(([0, 1], [1, 2]))
    
    print(cg.nodes(data=True))
    print(cg.edges())
    print('-'*80)
    print(expected.nodes(data=True))
    print(expected.edges())
    assert _equal_graphs(cg, expected)


def test_residue_crossing():
    '''
    Make sure we do cross residue boundaries and can rename residues
    '''
    mapping = {(0, 'C1'): [(0, 'B1')], (0, 'C2'): [(0, 'B1')], (0, 'C3'): [(0, 'B1')],
               (1, 'C1'): [(1, 'B1')], (1, 'C2'): [(1, 'B1')], (1, 'C3'): [(1, 'B1')],
               (2, 'C1'): [(2, 'B1')], (2, 'C2'): [(2, 'B1')], (2, 'C3'): [(2, 'B1')]}
    weights = {(0, 'B1'): {(0, 'C1'): 1, (0, 'C2'): 1, (0, 'C3'): 1, },
               (1, 'B1'): {(1, 'C1'): 1, (1, 'C2'): 1, (1, 'C3'): 1, },
               (2, 'B1'): {(2, 'C1'): 1, (2, 'C2'): 1, (2, 'C3'): 1, },}
    extra = ()
    mappings = {'universal': {'martini22': {'IPO_large': (mapping, weights, extra)}}}

    cg = do_mapping(AA_MOL, mappings, FF_MARTINI)

    expected = Molecule()
    expected.add_nodes_from((
        (0, {'resid': 1, 'resname': 'IPO_large', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
        (1, {'resid': 2, 'resname': 'IPO_large', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
        (2, {'resid': 3, 'resname': 'IPO_large', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
            ))
    expected.add_edges_from(([0, 1], [1, 2]))
    
    print(cg.nodes(data=True))
    print(cg.edges())
    print('-'*80)
    print(expected.nodes(data=True))
    print(expected.edges())
    assert _equal_graphs(cg, expected)


def _map_weights(mapping):
    inv_map = defaultdict(list)
    for from_, tos in mapping.items():
        for to in tos:
            inv_map[to].append(from_)
    weights = {}
    for bd, atoms in inv_map.items():
        weights[bd] = {atom: 1 for atom in atoms}
    return weights


def test_peptide():
    """
    Test multiple cg beads in residue
    """
    gly = {(0, 'C'): [(0, 'BB')], (0, 'N'): [(0, 'BB')], (0, 'O'): [(0, 'BB')], (0, 'CA'): [(0, 'BB')]}
    ile = {(0, 'C'): [(0, 'BB')], (0, 'N'): [(0, 'BB')], (0, 'O'): [(0, 'BB')], (0, 'CA'): [(0, 'BB')],
           (0, 'CB'): [(0, 'SC1')], (0, 'CG1'): [(0, 'SC1')], (0, 'CG2'): [(0, 'SC1')], (0, 'CD'): [(0, 'SC1')]}
    leu = {(0, 'C'): [(0, 'BB')], (0, 'N'): [(0, 'BB')], (0, 'O'): [(0, 'BB')], (0, 'CA'): [(0, 'BB')],
           (0, 'CB'): [(0, 'SC1')], (0, 'CG1'): [(0, 'SC1')], (0, 'CD1'): [(0, 'SC1')], (0, 'CD2'): [(0, 'SC1')]}
    extra = ()
    mappings = {'universal': {'martini22': {'GLY': (gly, _map_weights(gly), extra),
                                            'ILE': (ile, _map_weights(ile), extra),
                                            'LEU': (leu, _map_weights(leu), extra),}}}
    
    peptide = Molecule(force_field=FF_UNIVERSAL)
    aa = FF_UNIVERSAL.blocks['GLY'].to_molecule()
    peptide.merge_molecule(aa)

    aa = FF_UNIVERSAL.blocks['ILE'].to_molecule()
    peptide.merge_molecule(aa)

    aa = FF_UNIVERSAL.blocks['LEU'].to_molecule()
    peptide.merge_molecule(aa)

    peptide.add_edge(list(peptide.find_atoms(atomname='N', resid=1))[0],
                     list(peptide.find_atoms(atomname='C', resid=2))[0])
    peptide.add_edge(list(peptide.find_atoms(atomname='N', resid=2))[0],
                     list(peptide.find_atoms(atomname='C', resid=3))[0])

    for node in peptide:
        peptide.nodes[node]['atomid'] = node + 1
        peptide.nodes[node]['chain'] = ''

    cg = do_mapping(peptide, mappings, FF_MARTINI)

    expected = Molecule(force_field=FF_MARTINI)
    expected.add_nodes_from({1: {'atomname': 'BB',
                                 'atype': 'P5',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 1,
                                 'resid': 1,
                                 'resname': 'GLY'},
                             2: {'atomname': 'BB',
                                 'atype': 'P5',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 2,
                                 'resid': 2,
                                 'resname': 'ILE'},
                             3: {'atomname': 'SC1',
                                 'atype': 'AC1',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 2,
                                 'resid': 2,
                                 'resname': 'ILE'},
                             4: {'atomname': 'BB',
                                 'atype': 'P5',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 3,
                                 'resid': 3,
                                 'resname': 'LEU'},
                             5: {'atomname': 'SC1',
                                 'atype': 'AC1',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 3,
                                 'resid': 3,
                                 'resname': 'LEU'}}.items()
                            )
    expected.add_edges_from([(1, 2), (2, 3), (2, 4), (4, 5)])
    
    for node in expected:
        expected.nodes[node]['atomid'] = node + 1
    
    assert _equal_graphs(cg, expected)

if __name__ == '__main__':
    test_peptide()
