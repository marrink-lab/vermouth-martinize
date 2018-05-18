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

import copy
from vermouth.processors.do_mapping import do_mapping
import vermouth.forcefield
from vermouth.molecule import Molecule, Block
import networkx.algorithms.isomorphism as iso


FF_MARTINI = copy.copy(vermouth.forcefield.FORCE_FIELDS['martini22'])
FF_UNIVERSAL = copy.copy(vermouth.forcefield.FORCE_FIELDS['universal'])

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
block_aa = Block()
block_aa.name = 'IPO'
block_aa.add_nodes_from((
    ('C1', {'resid': 1, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C2', {'resid': 1, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C3', {'resid': 1, 'resname': 'IPO', 'atomname': 'C3'}),))
block_aa.add_edges_from([('C1', 'C2'), ('C2', 'C3')])

block_cg = Block()
block_cg.name = 'IPO'
block_cg.add_nodes_from((('B1', {'resid': 1, 'resname': 'IPO', 'atomname': 'B1'}),))

FF_MARTINI.blocks['IPO'] = block_cg
FF_UNIVERSAL.blocks['IPO'] = block_aa


def test_simple_aa_to_cg():
    mapping = {(0, 'C1'): [(0, 'B1')], (0, 'C2'): [(0, 'B1')], (0, 'C3'): [(0, 'B1')]}
    weights = {(0, 'B1'): {(0, 'C1'): 1, (0, 'C2'): 1, (0, 'C3'): 1, }}
    extra = ()
    mappings = {'universal': {'martini22': {'IPO': (mapping, weights, extra)}}}

    cg = do_mapping(AA_MOL, mappings, FF_MARTINI)

    expected = Molecule()
    expected.add_nodes_from((
        (0, {'resid': 1, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
        (1, {'resid': 2, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 2}),
        (2, {'resid': 3, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 3}),
            ))
    expected.add_edges_from(([0, 1], [1, 2]))
    node_equal = iso.categorical_node_match(['resid', 'resname', 'atomname', 'chain', 'charge_group'], ['']*5)
    matcher = iso.GraphMatcher(cg, expected, node_match=node_equal)
    print(cg.nodes(data=True))
    print(cg.edges())
    print('-'*80)
    print(expected.nodes(data=True))
    print(expected.edges())
    assert matcher.is_isomorphic()


if __name__ == '__main__':
    test_simple_aa_to_cg()
