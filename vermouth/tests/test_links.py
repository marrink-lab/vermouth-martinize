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

import pytest
import numpy as np
from vermouth.processors import do_links, DoLinks
from vermouth.molecule import Molecule, Link
import vermouth.forcefield

@pytest.mark.parametrize(
    "orders, resids, answer", (
        # 0, 0
        ((0, 0), (4, 4), True),
        ((0, 0), (3, 4), False),
        # 0, n
        ((0, -3), (6, 3), True),
        ((0, -3), (6, 6), False),
        ((0, -3), (6, -2), False),
        ((0, -3), (6, 9), False),
        ((0, 3), (6, 9), True),
        ((0, 3), (6, 6), False),
        ((0, 3), (6, 3), False),
        # 0, ><
        ((0, '>'), (5, 8), True),
        ((0, '>'), (5, 5), False),
        ((0, '>'), (5, 3), False),
        ((0, '>>'), (5, 8), True),
        ((0, '>>'), (5, 5), False),
        ((0, '>>'), (5, 3), False),
        ((0, '<'), (5, 3), True),
        ((0, '<'), (5, 5), False),
        ((0, '<'), (5, 8), False),
        ((0, '<<'), (5, 3), True),
        ((0, '<<'), (5, 5), False),
        ((0, '<<'), (5, 8), False),
        # 0, *
        ((0, '*'), (4, 1), True),
        ((0, '*'), (4, 8), True),
        ((0, '*'), (4, 4), False),
        ((0, '**'), (4, 1), True),
        ((0, '**'), (4, 8), True),
        ((0, '**'), (4, 4), False),
        # n, 0
        ((4, 0), (6, 2), True),
        ((4, 0), (6, 8), False),
        ((-4, 0), (2, 6), True),
        ((-4, 0), (4, 6), False),
        # n, n
        ((5, 5), (7, 7), True),
        ((5, 5), (7, 8), False),
        ((5, 7), (2, 4), True),
        ((5, 7), (2, 5), False),
        ((5, 7), (2, 1), False),
        ((7, 5), (4, 2), True),
        ((7, 5), (2, 5), False),
        ((7, 5), (2, 1), False),
        ((-2, -3), (5, 4), True),
        ((-2, -3), (5, 8), False),
        # n, ><  no comparison
        ((3, '>'), (8, 9), True),
        ((3, '>'), (8, 8), True),
        ((3, '>'), (8, 5), True),
        ((3, '>>'), (8, 9), True),
        ((3, '>>'), (8, 8), True),
        ((3, '>>'), (8, 5), True),
        ((3, '<'), (8, 6), True),
        ((3, '<'), (8, 8), True),
        ((3, '<'), (8, 9), True),
        ((3, '<<'), (8, 6), True),
        ((3, '<<'), (8, 8), True),
        ((3, '<<'), (8, 9), True),
        # n, *  no comparison
        ((2, '*'), (5, 6), True),
        ((2, '*'), (5, 3), True),
        ((2, '*'), (5, 5), True),
        ((2, '**'), (5, 6), True),
        ((2, '**'), (5, 3), True),
        ((2, '**'), (5, 5), True),
        # ><, 0
        (('>', 0), (6, 3), True),
        (('>', 0), (3, 3), False),
        (('>', 0), (3, 6), False),
        (('>>', 0), (6, 3), True),
        (('>>', 0), (3, 3), False),
        (('>>', 0), (3, 6), False),
        (('<', 0), (3, 6), True),
        (('<', 0), (3, 3), False),
        (('<', 0), (6, 3), False),
        (('<<', 0), (3, 6), True),
        (('<<', 0), (3, 3), False),
        (('<<', 0), (6, 3), False),
        # ><, n  no comparison
        (('>', 3), (8, 9), True),
        (('>', 3), (8, 8), True),
        (('>', 3), (8, 5), True),
        (('>>', 3), (8, 9), True),
        (('>>', 3), (8, 8), True),
        (('>>', 3), (8, 5), True),
        (('<', 3), (8, 6), True),
        (('<', 3), (8, 8), True),
        (('<', 3), (8, 9), True),
        (('<<', 3), (8, 6), True),
        (('<<', 3), (8, 8), True),
        (('<<', 3), (8, 9), True),
        # ><, ><
        (('>', '>'), (3, 3), True),
        (('>', '>'), (3, 4), False),
        (('>', '>'), (3, 1), False),
        (('>>>', '>>>'), (3, 3), True),
        (('>>>', '>>>'), (3, 4), False),
        (('>>>', '>>>'), (3, 1), False),
        (('<', '<'), (3, 3), True),
        (('<', '<'), (3, 4), False),
        (('<', '<'), (3, 1), False),
        (('<<<', '<<<'), (3, 3), True),
        (('<<<', '<<<'), (3, 4), False),
        (('<<<', '<<<'), (3, 1), False),
        (('>', '>>'), (3, 6), True),
        (('>', '>>'), (3, 3), False),
        (('>', '>>'), (3, 1), False),
        (('>>', '>>>'), (3, 6), True),
        (('>>', '>>>'), (3, 3), False),
        (('>>', '>>>'), (3, 1), False),
        (('<', '<<'), (5, 3), True),
        (('<', '<<'), (5, 5), False),
        (('<', '<<'), (5, 7), False),
        (('>>', '<<<'), (5, 3), True),
        (('>>', '<<<'), (5, 5), False),
        (('<<', '>>>'), (3, 5), True),
        (('<<', '>>>'), (3, 3), False),
        (('<<', '>>>'), (3, 1), False),
        # ><, *  no comparison
        (('>', '*'), (3, 3), True),
        (('>>', '***'), (3, 2), True),
        (('<<<', '**'), (2, 5), True),
        # *, 0
        (('***', 0), (4, 5), True),
        (('**', 0), (5, 4), True),
        (('****', 0), (5, 5), False),
        # *, n  no comparison
        (('***', 1), (4, 5), True),
        (('**', 2), (5, 4), True),
        (('****', -3), (5, 5), True),
        # *, ><  no comparison
        (('***', '<<'), (4, 5), True),
        (('**', '>'), (5, 4), True),
        (('****', '>>>'), (5, 5), True),
        # *, *
        (('*', '*'), (6, 6), True),
        (('*', '*'), (6, 5), False),
        (('**', '**'), (6, 6), True),
        (('**', '**'), (6, 5), False),
        (('**', '****'), (5, 7), True),
        (('**', '****'), (5, 5), False),
    )
)
def test_match_order(orders, resids, answer):
    order1, order2 = orders
    resid1, resid2 = resids
    match = do_links.match_order(order1, resid1, order2, resid2)
    assert match == answer


@pytest.mark.parametrize("order", (
    1.2, -3.9, None, True, False, {'a': 3},
    '><', '#>>', '#$H', '', [], np.nan, 
))
def test_order_errors(order):
    with pytest.raises(ValueError):
        do_links._interpret_order(order)


@pytest.mark.parametrize("order, ref_order_type, ref_order_value", (
    (0, 'number', 0),
    (2, 'number', 2),
    (-3, 'number', -3),
    ('>', '><', 1),
    ('>>', '><', 2),
    ('>>>', '><', 3),
    ('<', '><', -1),
    ('<<', '><', -2),
    ('<<<', '><', -3),
    ('*', '*', 1),
    ('**', '*', 2),
    ('***', '*', 3),
))
def test_interpret_order(order, ref_order_type, ref_order_value):
    order_type, order_value = do_links._interpret_order(order)
    assert order_type == ref_order_type
    assert order_value == ref_order_value

def make_mol(mol_nodes, mol_edges=[], **kwargs):
    mol = Molecule(**kwargs)
    mol.add_nodes_from(mol_nodes)
    mol.add_edges_from(mol_edges)
    return mol

def make_link(mol_nodes, mol_edges=[]):
    mol = Link()
    mol.add_nodes_from(mol_nodes)
    mol.add_edges_from(mol_edges)
    return mol


@pytest.mark.parametrize('mol_nodes, mol_edges, link_nodes, link_edges, expected_nodes, expected_edges', (
    (
        [(0, {'atomname': 'a'}), (1, {'atomname': 'b'}), (2, {'atomname': 'c'})],
        [(0, 1), (1, 2)],
        [[(0, {'atomname': 'a'}), (1, {'atomname': 'b'}), (2, {'atomname': 'c', 'replace': {'atomname': None}})]],
        [[(0, 1), (1, 2)]],
        [(0, {'atomname': 'a'}), (1, {'atomname': 'b'})],
        [(0, 1)],
    ),
    (
        [(0, {'atomname': 'a'}), (1, {'atomname': 'b'})],
        [(0, 1)],
        [[(0, {'atomname': 'a', 'replace': {'atomname': 'a1'}}), (1, {'atomname': 'b', 'replace': {'atomname': None}})]],
        [[(0, 1)]],
        [(0, {'atomname': 'a1'})],
        [],
    ),
    (
        [(0, {'atomname': 'a'}), (1, {'atomname': 'b'}), (2, {'atomname': 'a'})],
        [(0, 1), (1, 2)],
        [[(0, {'atomname': 'a', 'replace': {'atomname': 'a1'}}), (1, {'atomname': 'b', 'replace': {'atomname': None}})]],
        [[(0, 1)]],
        [(0, {'atomname': 'a1'}), (2, {'atomname': 'a1'})],
        [],
    ),
    (
        [(0, {'atomname': 'a'}), (1, {'atomname': 'b'}), (2, {'atomname': 'c'})],
        [(0, 1), (1, 2)],
        [
            [(0, {'atomname': 'a', 'replace': {'atomname': 'a1'}}), (1, {'atomname': 'b', 'replace': {'atomname': None}})],
            [(0, {'atomname': 'a1', 'replace': {'atomname': 'a2'}}), (1, {'atomname': 'b'})]
        ],
        [[(0, 1)], [(0, 1)]],
        [(0, {'atomname': 'a1'}), (2, {'atomname': 'c'})],
        [],
    ),
    (
        [(0, {'atomname': 'a'}), (1, {'atomname': 'b'}), (2, {'atomname': 'c'})],
        [(0, 1), (1, 2)],
        [
            [(0, {'atomname': 'a', 'replace': {'atomname': 'a1'}}), (1, {'atomname': 'b', 'replace': {'atomname': 'b1'}})],
            [(0, {'atomname': 'c', 'replace': {'atomname': 'c1'}}), (1, {'atomname': 'b1', 'replace': {'atomname': 'b2'}})]
        ],
        [[(0, 1)], [(0, 1)]],
        [(0, {'atomname': 'a1'}), (1, {'atomname': 'b2'}), (2, {'atomname': 'c1'})],
        [(0, 1), (1, 2)],
    ),
))
def test_link_processor(mol_nodes, mol_edges, link_nodes, link_edges,
                        expected_nodes, expected_edges):
    links = [make_link(nodes, edges) for nodes, edges in zip(link_nodes, link_edges)]
    ff = vermouth.forcefield.get_native_force_field('universal')
    ff.links = links

    mol = make_mol(mol_nodes, mol_edges, force_field=ff)

    out = DoLinks().run_molecule(mol)
    assert dict(out.nodes(data=True)) == dict(expected_nodes)
    assert set(out.edges(data=False)) == set(expected_edges)
