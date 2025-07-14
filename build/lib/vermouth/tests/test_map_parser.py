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
Unit tests for the mapping file parser and its utilities.
"""
import textwrap

import networkx as nx
import pytest

from vermouth.forcefield import ForceField
from vermouth.molecule import Block, Link
from vermouth.map_parser import MappingDirector, MappingBuilder
from vermouth.tests.helper_functions import equal_graphs


# PyLint does *not* like pytest fixtures...
# pylint: disable=redefined-outer-name

@pytest.fixture
def force_fields():
    """
    Creates a simple forcefield for tests
    """
    ffa = ForceField(name='test_A')
    block_a = Block(name='block_A', force_field=ffa)
    block_a.add_nodes_from(['A1', 'A2', 'B1', 'B2'])
    for node in block_a.nodes:
        block_a.nodes[node]['atomname'] = node
        block_a.nodes[node]['resname'] = 'block_A'
    block_a.add_edges_from([('A1', 'A2'), ('B1', 'B2')])
    ffa.blocks['block_A'] = block_a
    ffa.modifications['block_A'] = Link(block_a)
    block_b = Block(name='block_B', force_field=ffa)
    block_b.add_node('A', atomname='A', resname='block_B', replace={'atomname': 'B'})
    ffa.blocks['block_B'] = block_b
    return {'test_A': ffa}


def equal_graph_data(found_nodes, found_edges, expected_nodes, expected_edges):
    """
    Returns True iff the data provided produces two equivalent graphs.
    """
    found_graph = nx.Graph()
    found_graph.add_nodes_from(found_nodes)
    found_graph.add_edges_from(found_edges)
    exp_graph = nx.Graph()
    exp_graph.add_nodes_from(expected_nodes)
    exp_graph.add_edges_from(expected_edges)
    return equal_graphs(found_graph, exp_graph, node_attrs=None, edge_attrs=None)


@pytest.fixture
def director(force_fields):
    """
    Produces a MappinDirector for parsing.
    """
    return MappingDirector(force_fields=force_fields, builder=MappingBuilder())


@pytest.mark.parametrize('lines, expected', (
    (  # Simple case
        """
        [modification]
        [ to ]
        martini22

        [ from ]
        universal
        """,
        {
            'ff_from': 'universal',
            'ff_to': 'martini22'
        }
    ),
    (  # Test space significance on section headers
        """
        [block ]
        [from ]
        martini30
        [ to]
        universal
        """,
        {
            'ff_from': 'martini30',
            'ff_to': 'universal'
        }
    ),
    (  # Test comments on values
        """
        [block ]
        [from ]
        martini30; comment
        [ to]
        universal ; more comment
        """,
        {
            'ff_from': 'martini30',
            'ff_to': 'universal'
        }
    ),
    (  # Also test comments on section headers
        """
        [block ] ; more comment
        [from ]; comment
        martini30
        [ to]       ; even more comments
        universal
        """,
        {
            'ff_from': 'martini30',
            'ff_to': 'universal'
        }
    ),
    (  # Test simplest to nodes
        """
        [block]
        [ to blocks]
        !B {}
        [ to nodes]
        B:B {"boing": "a"}
        """,
        {
            'block_to': ([(0, {'atomname': 'B', "boing": 'a'})], []),
        }
    ),
    (  # Test to nodes with multiple nodes and block attributes
        """
        [block]
        [ to blocks]
        !B {"plop": 42}
        [ to nodes]
        B:B {"boing": "a"}
        C {}
        """,
        {
            'block_to': ([(0, {'atomname': 'B', "boing": 'a', "plop": 42}),
                          (1, {'atomname': 'C', 'plop': 42})],
                         []),
        }
    ),
    (  # to nodes with identical atomnames
        """
        [block]
        [ to blocks]
        !B {}
        [ to nodes]
        B {"boing": "a"}
        B {}
        """,
        {
            'block_to': ([(0, {'atomname': 'B', "boing": 'a'}),
                          (1, {'atomname': 'B'})],
                         []),
        }
    ),
    (  # to edges, with edge attribute
        """
        [block]
        [ to blocks]
        !B {}
        [ to nodes]
        B {"boing": "a"}
        C
        [ to edges ]
        B:C B:B {"peanut": "butter"}
        """,
        {
            'block_to': ([(0, {'atomname': 'B', "boing": 'a'}),
                          (1, {'atomname': 'C'})],
                         [(0, 1, {"peanut": "butter"})]),
        }
    ),
    (  # Several block IDs for nodes with identical atomnames
        """
        [block]
        [ to blocks]
        !B1 {"resid": 1}
        !B2 {"resid": 2}
        [ to nodes]
        B1:B {"boing": "a"}
        B2:B {}
        [ to edges ]
        B1:B B2:B
        """,
        {
            'block_to': ([(0, {'atomname': 'B', "boing": 'a', 'resid': 1}),
                          (1, {'atomname': 'B', "resid": 2})],
                         [(0, 1, {})]),
        }
    ),
    (  # Test nodes from and to, and mapping
        """
        [modification]
        [ from blocks ]
        !A1 {"ping": false}
        !A2 {"resid": 23}
        [ from nodes ]
        A1:A {"plop": 1}
        A2:A {}
        [ from edges ]
        A1:A A2:A
        [ to blocks]
        !B {}
        [ to nodes]
        B:B {"boing": "a"}
        [ mapping ] ; Mapping section is needed since unmapped from nodes get removed
        A1:A B
        A2:A B
        """,
        {
            'block_from': ([(0, {'atomname': 'A', "plop": 1, 'ping': False}),
                            (1, {'atomname': 'A', 'resid': 23})], [(0, 1, {})]),
            'block_to': ([(0, {'atomname': 'B', "boing": 'a'})], []),
        }
    ),
    (  # Identical to and from block IDs
        """
        [block]
        [ from blocks ]
        !A1 {"ping": false}
        !A2 {"resid": 23}
        [ from nodes ]
        A1:A {"plop": 1}
        A2:A {}
        [ from edges ]
        A1:A A2:A
        [ to blocks]
        !A1 {}
        [ to nodes]
        B {"boing": "a"}
        [ mapping ]
        A1:A B
        A2:A B
        """,
        {
            'block_from': ([(0, {'atomname': 'A', "plop": 1, 'ping': False}),
                            (1, {'atomname': 'A', 'resid': 23})], [(0, 1, {})]),
            'block_to': ([(0, {'atomname': 'B', "boing": 'a'})], []),
            'mapping': {0: {0: 1}, 1: {0: 1}},
            'references': {}
        }

    ),
    (  # Reference atoms and mapping weights
        """
        [modification]
        [ to ]
        ff_a
        [ from]
        ff_b
        [ from blocks ]
        !A1 {"ping": false}
        !A2 {"resid": 23}
        [ from nodes ]
        A1:A {"plop": 1}
        A2:A {}
        [ from edges ]
        A1:A A2:A
        [ to blocks]
        !B {}
        [ to nodes]
        B:B {"boing": "a"}
        [ mapping ]
        A1:A B
        A2:A B 2
        [ reference atoms ]
        B A2:A
        """,
        {
            'block_from': ([(0, {'atomname': 'A', "plop": 1, 'ping': False}),
                            (1, {'atomname': 'A', 'resid': 23})], [(0, 1, {})]),
            'block_to': ([(0, {'atomname': 'B', "boing": 'a'})], []),
            'ff_to': 'ff_a',
            'ff_from': 'ff_b',
            'references': {0: 1},
            'mapping': {0: {0: 1}, 1: {0: 2}}
        }
    ),
    (  # Block shorthand, one node mapping to multiple others
        """
        [ block ]
        [ from blocks ]
        !ALA#1 !GLY#2
        [ to blocks ]
        !A#5 !B13 !C1#3
        [ from nodes ]
        ALA#1:A
        B
        GLY#2:A
        [ to nodes ]
        A#5:A
        B13:A
        B13:B
        C1#3:A
        [ mapping ]
        ALA#1:A A#5:A
        ALA#1:A B13:A
        B       B13:B
        GLY#2:A C1#3:A
        """,
        {
            'block_from': ([(0, {"resname": 'ALA', 'resid': 1, 'atomname': 'A'}),
                            (1, {"resname": 'ALA', 'resid': 1, 'atomname': 'B'}),
                            (2, {"resname": 'GLY', 'resid': 2, 'atomname': 'A'})],
                           []),
            'block_to': ([(0, {"resname": 'A', 'resid': 5, 'atomname': 'A'}),
                          (1, {"resname": 'B13', 'resid': 1, 'atomname': 'A'}),
                          (2, {"resname": 'B13', 'resid': 1, 'atomname': 'B'}),
                          (3, {"resname": 'C1', 'resid': 3, 'atomname': 'A'})],
                         []),
            'mapping': {0: {0: 1, 1: 1}, 1: {2: 1}, 2: {3: 1}}
        }
    ),
    (  # Block shorthand, remove unmapped nodes
        """
        [ block ]
        [ from blocks ]
        !ALA#1 !GLY#2
        [ to blocks ]
        !A#5 !B13 !C1#3
        [ from nodes ]
        ALA#1:A
        B
        GLY#2:A
        [ to nodes ]
        A#5:A
        B13:A
        B13:B
        C1#3:A
        [ mapping ]
        ALA#1:A A#5:A
        ALA#1:A B13:A
        GLY#2:A C1#3:A
        """,
        {
            'block_from': ([(0, {"resname": 'ALA', 'resid': 1, 'atomname': 'A'}),
                            (2, {"resname": 'GLY', 'resid': 2, 'atomname': 'A'})],
                           []),
            'block_to': ([(0, {"resname": 'A', 'resid': 5, 'atomname': 'A'}),
                          (1, {"resname": 'B13', 'resid': 1, 'atomname': 'A'}),
                          (2, {"resname": 'B13', 'resid': 1, 'atomname': 'B'}),
                          (3, {"resname": 'C1', 'resid': 3, 'atomname': 'A'})],
                         []),
            'mapping': {0: {0: 1, 1: 1}, 2: {3: 1}}
        }
    ),
    (  # Block shorthand, fetch single block
        """
        [ block ]
        [ from ]
        test_A
        [ from blocks ]
        block_A
        [ to blocks ]
        !A
        [ to nodes ]
        A
        [ mapping ]
        A1 A
        A2 A
        B1 A
        B2 A
        """,
        {
            'block_from': ([(0, {"resname": 'block_A', 'resid': 1,
                                 'atomname': 'A1', 'charge_group': 1}),
                            (1, {"resname": 'block_A', 'resid': 1,
                                 'atomname': 'A2', 'charge_group': 1}),
                            (2, {"resname": 'block_A', 'resid': 1,
                                 'atomname': 'B1', 'charge_group': 1}),
                            (3, {"resname": 'block_A', 'resid': 1,
                                 'atomname': 'B2', 'charge_group': 1})],
                           [(0, 1, {}), (2, 3, {})]),
            'block_to': ([(0, {"resname": 'A', 'resid': 1, 'atomname': 'A'})],
                         []),
            'mapping': {0: {0: 1}, 1: {0: 1}, 2: {0: 1}, 3: {0: 1}}
        }
    ),
    (  # Block shorthand, fetch single modification
        """
        [ modification ]
        [ from ]
        test_A
        [ from blocks ]
        block_A
        [ from nodes ]
        C
        [ to blocks ]
        !A
        [ to nodes ]
        A
        [ mapping ]
        A1 A
        A2 A
        B1 A
        B2 A
        C  A
        """,
        {
            'block_from': ([(0, {"resname": 'block_A', 'resid': 1,
                                 'atomname': 'A1', 'charge_group': 1}),
                            (1, {"resname": 'block_A', 'resid': 1,
                                 'atomname': 'A2', 'charge_group': 1}),
                            (2, {"resname": 'block_A', 'resid': 1,
                                 'atomname': 'B1', 'charge_group': 1}),
                            (3, {"resname": 'block_A', 'resid': 1,
                                 'atomname': 'B2', 'charge_group': 1}),
                            (4, {'resid': 1, 'atomname': 'C'})],
                           [(0, 1, {}), (2, 3, {})]),
            'block_to': ([(0, {'resid': 1, 'atomname': 'A'})],
                         []),
            'mapping': {0: {0: 1}, 1: {0: 1}, 2: {0: 1}, 3: {0: 1}, 4: {0: 1}}
        }
    ),
    (  # Block longhand, fetch multiple
        """
        [ block ]
        [ to ]
        test_A
        [ to blocks ]
        block_A#1 {"resname": "block_A", "resid": 1}
        blockA {"resname": "block_A", "resid": 2}
        [ from blocks ]
        !A
        [ from nodes ]
        A
        [ mapping ]
        A block_A#1:A1
        A A2
        A B1
        A B2
        A blockA:A1
        A A2
        A B1
        A B2
        """,
        {
            'block_to': ([(0, {"resname": 'block_A', 'resid': 1,
                               'atomname': 'A1', 'charge_group': 1}),
                          (1, {"resname": 'block_A', 'resid': 1,
                               'atomname': 'A2', 'charge_group': 1}),
                          (2, {"resname": 'block_A', 'resid': 1,
                               'atomname': 'B1', 'charge_group': 1}),
                          (3, {"resname": 'block_A', 'resid': 1,
                               'atomname': 'B2', 'charge_group': 1}),
                          (4, {"resname": 'block_A', 'resid': 2,
                               'atomname': 'A1', 'charge_group': 2}),
                          (5, {"resname": 'block_A', 'resid': 2,
                               'atomname': 'A2', 'charge_group': 2}),
                          (6, {"resname": 'block_A', 'resid': 2,
                               'atomname': 'B1', 'charge_group': 2}),
                          (7, {"resname": 'block_A', 'resid': 2,
                               'atomname': 'B2', 'charge_group': 2})],
                         [(0, 1, {}), (2, 3, {}), (4, 5, {}), (6, 7, {})]),
            'block_from': ([(0, {"resname": 'A', 'resid': 1, 'atomname': 'A'})],
                           []),
            'mapping': {0: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}}
        }
    ),
    (  # Block shorthand, fetch single from block with replace attr
       # Behaviour changed in #440 to use atomname *before* replacement
        """
        [ block ]
        [ from ]
        test_A
        [ from blocks ]
        block_B
        [ to blocks ]
        !A
        [ to nodes ]
        A
        [ mapping ]
        A A
        """,
        {
            'block_from': ([(0, {"resname": 'block_B', 'resid': 1,
                                 'atomname': 'B', 'charge_group': 1})],
                           []),
            'block_to': ([(0, {"resname": 'A', 'resid': 1, 'atomname': 'A'})],
                         []),
            'mapping': {0: {0: 1}}
        }
    ),
    (  # Block shorthand, fetch single to block with replace attr
        """
        [ block ]
        [ to ]
        test_A
        [ to blocks ]
        block_B
        [ from blocks ]
        !A
        [ from nodes ]
        A
        [ mapping ]
        A A
        """,
        {
            'block_to': ([(0, {"resname": 'block_B', 'resid': 1,
                               'atomname': 'A', 'charge_group': 1,
                               'replace': {'atomname': 'B'}})],
                         []),
            'block_from': ([(0, {"resname": 'A', 'resid': 1, 'atomname': 'A'})],
                           []),
            'mapping': {0: {0: 1}}
        }
    ),
))
def test_single_mapping_attrs(director, lines, expected):
    """
    Tests that when a file defines a single mapping the produced mapping has
    the correct attributes.
    """
    lines = textwrap.dedent(lines).splitlines()
    mappings = list(director.parse(lines))
    assert len(mappings) == 1
    mapping = mappings[0]
    for attr_name, val in expected.items():
        if 'block' in attr_name:
            nodes, edges = val
            equal_graph_data(list(getattr(mapping, attr_name).nodes(data=True)),
                             list(getattr(mapping, attr_name).edges(data=True)),
                             nodes, edges)
        else:
            assert getattr(mapping, attr_name) == val, attr_name


@pytest.mark.parametrize('map_type', (
    '[block]',
    '[modification]'
))
@pytest.mark.parametrize('number', (
    1,
    2,
    3
))
def test_multiple_mappings(director, map_type, number):
    """
    Tests that when a single file defines multiple mappings it produces the
    correct number of mappings.
    """
    lines = (map_type + '\n') * number
    lines = lines.splitlines()
    mappings = list(director.parse(lines))
    assert len(mappings) == number
