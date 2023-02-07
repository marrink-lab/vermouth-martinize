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
import itertools

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
import vermouth
import vermouth.molecule
import vermouth.forcefield
from vermouth.molecule import Interaction, Molecule

from .molecule_strategies import (random_molecule, random_block, random_link,
                                  SANE_NAME_STRATEGY,)

# pylint: disable=redefined-outer-name, no-value-for-parameter


# In some cases, drawing from hypothesis.strategies.text is extremely slow and
# triggers the heath check warnings. It appears to be due to some internal
# combinations of timers and cache. This is a workwround that forces the
# generation, and the caching, of the character table out of any test. This
# way, the table is cached, and the time required to build it is not accounted
# in the health checks.
# See <https://github.com/HypothesisWorks/hypothesis/issues/1153>.
hypothesis.internal.charmap.charmap()


@pytest.fixture
def molecule():
    molecule = vermouth.molecule.Molecule()
    molecule.meta['test'] = True
    molecule.meta['test_mutable'] = [0, 1, 2]
    # The node keys should not be in a sorted order as it would mask any issue
    # due to the keys being accidentally sorted.
    molecule.add_node(2, atomname='CC')
    molecule.add_node(0, atomname='AA', mutable=[7, 8, 9])
    molecule.add_node(1, atomname='BB')
    molecule.add_edge(0, 1)
    molecule.add_edge(0, 2)
    molecule.add_interaction(
        type_='bonds',
        atoms=(0, 1),
        parameters=['1', '2'],
        meta={'unmutable': 0, 'mutable': [4, 5, 6]},
    )
    molecule.add_interaction(
        type_='bonds',
        atoms=(0, 2),
        parameters=['a', 'b'],
    )
    return molecule


@pytest.fixture
def molecule_copy(molecule):
    return molecule.copy()


@pytest.mark.parametrize('atoms, bonds, interactions, removed, expected', [
    # empty molecule
    ([], [], [], [], {}),
    # Nodes but no interactions
    ([1, 2, 3], [(1, 2), (2, 3)], [], [2], {}),
    # interactions that all need to be removed
    ([1, 2], [(1, 2)], [('bond', (1, 2), {})], [2], {}),
    # Molecule with interactions of which some need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4), (5, 7)],
        [('bond', (5, 4), {}), ('bond', (1, 2), {})],
        [5],
        {'bond': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})]}
    ),
    # Molecule with interactions of which none need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4)],
        [('bond', (5, 4), {}), ('bond', (1, 2), {})],
        [6, 7],
        {'bond': [vermouth.molecule.Interaction(atoms=(5, 4), meta={}, parameters={}),
                  vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})]}
    ),
    # Molecule with interactions of different types of which all need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4), (5, 7), (6, 7)],
        [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})],
        [1, 5],
        {}
    ),
    # Molecule with interactions of different types of which some need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4), (5, 7), (6, 7)],
        [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})],
        [5],
        {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})],
         'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={})]}
    ),
    # Molecule with interactions of different types of which none need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4), (5, 7), (6, 7)],
        [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})],
        [3],
        {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})],
         'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={}),
                  vermouth.molecule.Interaction(atoms=(5, 4), meta={}, parameters={})]}
    ),
])
def test_remove_nodes_from(atoms, bonds, interactions, removed, expected):
    """
    Test whether molecule.remove_nodes_from also removes the corresponding
    interactions
    """
    molecule = vermouth.molecule.Molecule()
    molecule.add_nodes_from(atoms)
    molecule.add_edges_from(bonds)
    for type_, atoms, params in interactions:
        molecule.add_interaction(type_, atoms, params)

    molecule.remove_nodes_from(removed)
    assert molecule.interactions == expected


@pytest.mark.parametrize('atoms, bonds, interactions, removed, expected', [
    # empty molecule
    ([], [], [], None, {}),
    # Nodes but no interactions
    ([1, 2, 3], [(1, 2), (2, 3)], [], 2, {}),
    # interactions that all need to be removed
    ([1, 2], [(1, 2)], [('bond', (1, 2), {})], 2, {}),
    # Molecule with interactions of which some need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4), (5, 7)],
        [('bond', (5, 4), {}), ('bond', (1, 2), {})],
        5,
        {'bond': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})]}
    ),
    # Molecule with interactions of different types of which some need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4), (5, 7), (6, 7)],
        [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})],
        5,
        {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})],
         'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={})]}
    ),
    # Molecule with interactions of different types of which none need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4), (5, 7), (6, 7)],
        [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})],
        3,
        {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})],
         'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={}),
                  vermouth.molecule.Interaction(atoms=(5, 4), meta={}, parameters={})]}
    ),
    (
        [1, 2, 3, 4],
        [(1, 2), (2, 3), (3, 4)],
        [('bond', (1, 2), {}), ('bond', (2, 3), {}), ('bond', (3, 4), {})],
        2,
        {'bond': [vermouth.molecule.Interaction(atoms=(3, 4), meta={}, parameters={})]},
    ),
])
def test_remove_node(atoms, bonds, interactions, removed, expected):
    """
    Test whether molecule.remove_node also removes the corresponding
    interactions
    """
    molecule = vermouth.molecule.Molecule()
    molecule.add_nodes_from(atoms)
    molecule.add_edges_from(bonds)
    for type_, atoms, params in interactions:
        molecule.add_interaction(type_, atoms, params)

    if removed:
        molecule.remove_node(removed)

    assert molecule.interactions == expected

@pytest.fixture
def molecule_subgraph(molecule):
    return molecule.subgraph([2, 0])


def test_copy(molecule, molecule_copy):
    assert molecule_copy is not molecule
    assert molecule_copy.meta == molecule.meta
    assert list(molecule_copy.nodes) == list(molecule.nodes)
    assert list(molecule_copy.nodes.values()) == list(molecule.nodes.values())
    assert molecule_copy.interactions == molecule.interactions


def test_copy_meta_mod(molecule, molecule_copy):
    molecule_copy.meta['test'] = False
    assert molecule_copy.meta['test'] != molecule.meta['test']
    # We are doing a copy, not a deep copy.
    assert molecule_copy.meta['test_mutable'] is molecule.meta['test_mutable']


def test_copy_node_mod(molecule, molecule_copy):
    molecule_copy.nodes[0]['atomname'] = 'mod'
    assert molecule_copy.nodes[0]['atomname'] != molecule.nodes[0]['atomname']
    extra_value = 'a new attribute'
    molecule_copy.nodes[0]['extra'] = extra_value
    assert molecule_copy.nodes[0]['extra'] == extra_value
    assert 'extra' not in molecule.nodes

    # We are looking at a copy, not a deep copy
    assert molecule_copy.nodes[0]['mutable'] is molecule.nodes[0]['mutable']

    molecule_copy.add_node('new')
    assert 'new' in molecule_copy.nodes
    assert 'new' not in molecule


def test_copy_edge_mod(molecule, molecule_copy):
    molecule_copy.add_edge(1, 2)
    assert (1, 2) in molecule_copy.edges
    assert (1, 2) not in molecule.edges
    molecule_copy.edges[(0, 1)]['attribute'] = 1
    assert molecule_copy.edges[(0, 1)]['attribute'] == 1
    assert 'attribute' not in molecule.edges[(0, 1)]


def test_copy_interactions_mod(molecule, molecule_copy):
    molecule_copy.add_interaction(
        type_='bonds',
        atoms=(0, 2),
        parameters=['3', '4'],
        meta={'unmutable': 0},
    )
    n_bonds = len(molecule.interactions['bonds'])
    n_bonds_copy = len(molecule_copy.interactions['bonds'])
    assert n_bonds_copy > n_bonds


def test_subgraph_base(molecule_subgraph):
    assert tuple(molecule_subgraph) == (2, 0)  # order matters!
    assert (0, 2) in molecule_subgraph.edges
    assert (0, 1) not in molecule_subgraph.edges  # node 1 is not there


def test_subgraph_interactions(molecule_subgraph):
    bond_atoms = [bond.atoms for bond in molecule_subgraph.interactions['bonds']]
    assert (0, 2) in bond_atoms
    assert (0, 1) not in bond_atoms


def test_link_predicate_match():
    lp = vermouth.molecule.LinkPredicate(None)
    with pytest.raises(NotImplementedError):
        lp.match(1, 2)


@pytest.fixture
def edges_between_molecule():
    """
    Build an empty molecule with known connectivity.

    The molecule does not have any node attribute nor any molecule metadata. It
    only has a bare graph with a few nodes and edges.

    The graph looks like::

        0 - 1 - 3 - 4 - 5 - 7 - 8     9 - 10 - 11 - 12
            |           |
            2           6

    """
    molecule = vermouth.molecule.Molecule()
    molecule.add_edges_from((
        (0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (5, 6), (5, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
    ))
    for node1, node2, attributes in molecule.edges(data=True):
        attributes['arbitrary'] = '{} - {}'.format(min(node1, node2),
                                                   max(node1, node2))
    return molecule


@pytest.fixture
def edges_between_selections():
    """
    Build a static list of selections of nodes from :func:`edges_between_molecule`.
    """
    return [
        (0, 1, 2, 3),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (3, 4, 5, 6),
        (7, 8, 9, 10),
    ]


@pytest.mark.parametrize('data', [True, False])
@pytest.mark.parametrize('bunch1, bunch2, expected', (
    (0, 1, []), (0, 2, []), (0, 4, []), (1, 2, []), (2, 3, []),  # non-overlapping
    (0, 3, [(1, 3), (3, 4)]), (1, 3, [(4, 5), (5, 6), (5, 6), (5, 7)]),
    (1, 4, [(5, 7), (7, 8), (7, 8)]), (2, 4, [(9, 10), (9, 10), (10, 11)]),
))
def test_edges_between(edges_between_molecule, edges_between_selections,
                       bunch1, bunch2, expected, data):
    """
    Test :meth:`vermouth.molecule.Molecule.edges_between`.
    """
    selection_1 = edges_between_selections[bunch1]
    selection_2 = edges_between_selections[bunch2]
    found = list(edges_between_molecule.edges_between(
        selection_1, selection_2, data=data
    ))
    sorted_found = sorted(sorted(edge[:2]) for edge in found)
    sorted_expected = sorted(sorted(edge) for edge in expected)
    assert sorted_found == sorted_expected
    if data:
        found_attributes = [
            edge[2]
            for edge in sorted(found, key=lambda x: x[:2])
        ]
        expected_attributes = [
            edges_between_molecule.edges[edge[0], edge[1]]
            for edge in sorted_expected
        ]
        assert found_attributes == expected_attributes



@pytest.mark.parametrize('selidx, expected', (
    (0, ((0, 1), (1, 2), (1, 3))),
    (1, ((5, 6), (5, 7), (7, 8))),
    (2, ((9, 10), (10, 11), (11, 12))),
    (3, ((3, 4), (4, 5), (5, 6))),
    (4, ((7, 8), (9, 10))),
))
def test_subgraph_edges(edges_between_molecule, edges_between_selections,
                        selidx, expected):
    """
    :meth:`vermouth.molecule.Molecule.subgraph` select the expected edges.

    See Also
    --------
    test_subgraph_base
        This test deals with a larger graph, but no metadata; while the graph
        in :func:`test_subgraph_base` uses a very small and simple selection
        but the graph has metadata.
    """
    subgraph = edges_between_molecule.subgraph(edges_between_selections[selidx])
    sorted_found = sorted(sorted(edge) for edge in subgraph.edges)
    sorted_expected = sorted(sorted(edge) for edge in expected)
    assert sorted_found == sorted_expected

    found_attributes = [
        subgraph.edges[edge[0], edge[1]]
        for edge in sorted_expected
    ]
    expected_attributes = [
        edges_between_molecule.edges[edge[0], edge[1]]
        for edge in sorted_expected
    ]
    assert found_attributes == expected_attributes


@pytest.mark.parametrize('left, right, expected', (
    (  # Same
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        True,
    ),
    (  # Difference in atoms
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'notC'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Difference in parameters
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['different'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Difference in meta
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0, 'other': True}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Equal with LinkParameterEffector
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        True,
    ),
    (  # Different arguments for LinkParameterEffector
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'C']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Different format_spec in LinkParameterEffector
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(
                                    ['A', 'B'], format_spec='.2f',
                                ),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Different LinkParameterEffector
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamAngle(['A', 'B', 'C']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Missing interaction category ("angles" is missing in one molecule)
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamAngle(['A', 'B', 'C']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamAngle(['A', 'B', 'C']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
        },
        False,
    ),
    (  # Empty interaction category
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamAngle(['A', 'B', 'C']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [],  # Empty!
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamAngle(['A', 'B', 'C']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
        },
        True,
    ),

))
def test_same_interactions(left, right, expected):
    """
    Test that Molecule.same_interactions works as expected.
    """
    left_mol = Molecule()
    left_mol.interactions = left
    right_mol = Molecule()
    right_mol.interactions = right
    assert left_mol.same_interactions(right_mol) == expected
    assert right_mol.same_interactions(left_mol) == expected


@pytest.mark.parametrize('left, right, expected', (
    (  # Simple identical
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        True,  # expected
    ),
    (  # Wrong order
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (1, {'c': (0, 1, 2), 'd': None}),
            (0, {'a': 'abc', 'b': 123}),
        ),
        False,  # expected
    ),

    (  # Different string
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'different', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,
    ),
    (  # Different number
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'abc', 'b': 900}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Different tuple
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (3, 2, 1), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Equal Numpy array
        (  # left
            (0, {'a': np.linspace(2, 5, num=7), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': np.linspace(2, 5, num=7), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        True,  # expected
    ),
    (  # Different Numpy array
        (  # left
            (0, {'a': np.linspace(2, 5, num=7), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': np.linspace(2, 8, num=7), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Different shaped Numpy array
        (  # left
            (0, {'a': np.linspace(2, 5, num=9), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': np.linspace(2, 5, num=9).reshape((3, 3)), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Mismatch types
        (  # left
            (0, {'a': np.linspace(2, 5, num=9), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'not an array', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Mismatch attribute key
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'abc', 'different': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # same dicts as value
        (  # left
            (0, {'a': {'a': 0, 'b': 1}}),
        ),
        (  # right
            (0, {'a': {'a': 0, 'b': 1}}),
        ),
        True,
    ),
    (  # different dicts as value
        (  # left
            (0, {'a': {'a': 0, 'b': 1}}),
        ),
        (  # right
            (0, {'a': {'a': None, 'b': 1}}),
        ),
        False,
    ),
))
def test_same_nodes(left, right, expected):
    left_mol = Molecule()
    left_mol.add_nodes_from(left)
    right_mol = Molecule()
    right_mol.add_nodes_from(right)
    assert left_mol.same_nodes(right_mol) == expected
    assert right_mol.same_nodes(left_mol) == expected


@pytest.mark.parametrize('effector_class', (
    vermouth.molecule.ParamDistance,
    vermouth.molecule.ParamAngle,
    vermouth.molecule.ParamDihedral,
    vermouth.molecule.ParamDihedralPhase,
))
@pytest.mark.parametrize('format_spec', (
    None, '.2f', '0.3f',
))
def test_link_parameter_effector_equal(effector_class, format_spec):
    """
    Test that equal LinkParameterEffector compare equal.
    """
    n_keys = effector_class.n_keys_asked
    left_keys = ['A{}'.format(idx) for idx in range(n_keys)]
    right_keys = copy.copy(left_keys)  # Let's be sure the id is different
    left = effector_class(left_keys, format_spec=format_spec)
    right = effector_class(right_keys, format_spec=format_spec)
    assert left == right


@pytest.mark.parametrize('effector_class', (
    vermouth.molecule.ParamDistance,
    vermouth.molecule.ParamAngle,
    vermouth.molecule.ParamDihedral,
    vermouth.molecule.ParamDihedralPhase,
))
@pytest.mark.parametrize('format_right, format_left', itertools.combinations(
    (None, '.2f', '0.3f', ), 2
))
def test_link_parameter_effector_diff_format(effector_class, format_left, format_right):
    """
    Test that two instances of LinkParameterEffector compare different if they
    have different formats.
    """
    n_keys = effector_class.n_keys_asked
    left_keys = ['A{}'.format(idx) for idx in range(n_keys)]
    right_keys = copy.copy(left_keys)  # Let's be sure the id is different
    left = effector_class(left_keys, format_spec=format_left)
    right = effector_class(right_keys, format_spec=format_right)
    assert left != right


@pytest.mark.parametrize('effector_class', (
    vermouth.molecule.ParamDistance,
    vermouth.molecule.ParamAngle,
    vermouth.molecule.ParamDihedral,
    vermouth.molecule.ParamDihedralPhase,
))
def test_link_parameter_effector_diff_keys(effector_class):
    """
    Test that LinkParameterEffector compare different if they have different keys.
    """
    n_keys = effector_class.n_keys_asked
    left_keys = ['A{}'.format(idx) for idx in range(n_keys)]
    right_keys = ['B{}'.format(idx) for idx in range(n_keys)]
    left = effector_class(left_keys)
    right = effector_class(right_keys)
    assert left != right


@pytest.mark.parametrize('left_class, right_class', itertools.combinations((
    vermouth.molecule.ParamDistance,
    vermouth.molecule.ParamAngle,
    vermouth.molecule.ParamDihedral,
    vermouth.molecule.ParamDihedralPhase,
), 2))
def test_link_parameter_effector_diff_class(left_class, right_class):
    """
    Test that LinkParameterEffector compare different if they have different classes.
    """
    left_n_keys = left_class.n_keys_asked
    left_keys = ['A{}'.format(idx) for idx in range(left_n_keys)]
    left = left_class(left_keys)

    right_n_keys = right_class.n_keys_asked
    right_keys = ['A{}'.format(idx) for idx in range(right_n_keys)]
    right = right_class(right_keys)

    assert left != right


@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
@hypothesis.given(random_molecule())
def test_molecule_equal(mol):
    """
    Two equal molecules are equal.
    """
    mol_copy = mol.copy()
    assert mol == mol_copy
    assert mol is not mol_copy


@hypothesis.given(random_block())
def test_block_equal(block):
    """
    Two equal blocks are equal.
    """
    block_copy = block.copy()
    assert block == block_copy
    assert block is not block_copy


@hypothesis.given(random_link(max_nodes=3, max_meta=3))
def test_link_equal(link):
    """
    Two equal links are equal.
    """
    link_copy = link.copy()
    assert link == link
    assert link is not link_copy


@pytest.mark.parametrize('left, right, expected', (
    (  # Equal edges, same order, same direction, same attributes
        [
            (0, 1, {'a': 0, 'b': [2, 3]}),
            (1, 3, {'foo': 'bar'}),
        ],
        [
            (0, 1, {'a': 0, 'b': [2, 3]}),
            (1, 3, {'foo': 'bar'}),
        ],
        True,
    ),
    (  # Equal edges, but different order
        [
            (0, 1, {'a': 0, 'b': [2, 3]}),
            (1, 3, {'foo': 'bar'}),
        ],
        [
            (1, 3, {'foo': 'bar'}),
            (0, 1, {'a': 0, 'b': [2, 3]}),
        ],
        True,
    ),
    (  # Equal edges, but different order and different direction
        [
            (0, 1, {'a': 0, 'b': [2, 3]}),
            (1, 3, {'foo': 'bar'}),
        ],
        [
            (3, 1, {'foo': 'bar'}),
            (0, 1, {'a': 0, 'b': [2, 3]}),
        ],
        True,
    ),
    (  # Edges only differ by an attribute
        [
            (0, 1, {'a': 0, 'b': [2, 3]}),
            (1, 3, {'foo': 'bar'}),
        ],
        [
            (0, 1, {'a': 0, 'b': [2, 40]}),
            (1, 3, {'foo': 'bar'}),
        ],
        False,
    ),
    (  # Edges are different
        [
            (0, 1, {'foo': 'bar'}),
            (2, 3, {}),
        ],
        [
            (5, 6, {'foo': 'bar'}),
            (3, 4, {}),
        ],
        False,
    ),
))
def test_same_edges(left, right, expected):
    """
    Compare edges between two molecules.
    """
    molecule_left = vermouth.molecule.Molecule()
    molecule_left.add_edges_from(left)
    molecule_right = vermouth.molecule.Molecule()
    molecule_right.add_edges_from(right)
    assert molecule_left.same_edges(molecule_right) == expected
    assert molecule_right.same_edges(molecule_left) == expected


@pytest.mark.parametrize('left, right, expected', (
    (  # Identical non-edges
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        True,
    ),
    (  # Identical non-edges, but disordered
        [
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
        ],
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        True,
    ),

    (  # Different number of non-edges
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('YY', {'klonk': 0}),
        ],
        False,
    ),
    (  # Same number of non-edges, but mismatched anchors
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        [
            ('AA', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('BB', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        False,
    ),
    (  # Mismatch in an attribute
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        [
            ('XX', {'foo': 'mismatch', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        False,
    ),
    (  # Mismatch in number of attributes, there is a shortcut in same_non_edges
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar'}),
            ('YY', {}),
        ],
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar', 'plop': 'else'}),
            ('YY', {'klonk': 0}),
        ],
        False,
    ),
    (
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar'}),
            ('XX', {'foo': 'no bar again'}),
            ('YY', {}),
        ],
        [
            ('XX', {'foo': 'bar', 'toto': [0, 1, 2]}),
            ('XX', {'foo': 'no bar'}),
            ('YY', {}),
            ('YY', {'something': 'tadaam'}),
        ],
        False,
    ),
))
def test_same_non_edges(left, right, expected):
    """
    Compare non-edges between two links.
    """
    link_left = vermouth.molecule.Link()
    link_left.non_edges = left
    link_right = vermouth.molecule.Link()
    link_right.non_edges = right
    assert link_left.same_non_edges(link_right) == expected
    assert link_right.same_non_edges(link_left) == expected


@pytest.mark.parametrize('interactions, expected', (
    (
        {},
        []
    ),
    (
        {'bonds': []},
        []
    ),
    (
        {'bonds': [Interaction(atoms=(1, 2), parameters=[], meta=[])]},
        ['bonds']
    ),
    (
        {'bonds': [Interaction(atoms=(1, 2), parameters=[], meta=[])],
         'angles': []},
        ['bonds']
    ),
    (
        {'bonds': [Interaction(atoms=(1, 2), parameters=[], meta=[])],
         'angles': [Interaction(atoms=(1, 2, 3), parameters=[], meta=[])]},
        ['bonds', 'angles']
    ),
    (
        {'angles': [Interaction(atoms=(1, 2, 3), parameters=[], meta=[])],
         'bonds': [Interaction(atoms=(1, 2), parameters=[], meta=[])],},
        ['bonds', 'angles']
    ),
    (
        {'angles': [Interaction(atoms=(1, 2, 3), parameters=[], meta=[])],
         'a_bonds': [Interaction(atoms=(10, 2), parameters=[], meta=[])],
         'bonds': [Interaction(atoms=(1, 2), parameters=[], meta=[])],},
        ['a_bonds', 'bonds', 'angles']
    ),
))
def test_interaction_sort(interactions, expected):
    """
    Test the order produced by Molecule._sort_interactions.
    """
    assert vermouth.molecule.Molecule.sort_interactions(interactions) == expected


@hypothesis.given(
    moltype=st.one_of(
        st.none(),
        SANE_NAME_STRATEGY,
    ),
    mol=random_molecule()
)
def test_str_method(mol, moltype):
    """
    Test Molecule.__str__
    """
    if moltype is not None:
        mol.meta['moltype'] = moltype
    else:
        moltype = 'molecule'
    found = str(mol)
    assert '{} with '.format(moltype) in found
    assert '{} {}'.format(len(mol), 'atoms') in found
    hypothesis.assume('' not in mol.interactions)
    for itype, interactions in mol.interactions.items():
        if interactions:
            assert '{} {}'.format(len(interactions), itype) in found
        else:
            assert '{} {}'.format(0, itype) not in found

def test_to_molecule():
    """
    Test if the to molecule function gives
    expected results.
    """
    test_block = vermouth.molecule.Block()
    test_block.add_edges_from([('A', 'B'), ('B', 'C')])
    test_block.interactions["bonds"] = [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),]

    test_molecule = test_block.to_molecule()

    ref_bonds = [Interaction(atoms=(0, 1),
                             parameters=['a', '0.2', '200'],
                             meta={'a': 0}),
                 Interaction(atoms=(1, 2),
                             parameters=['a', '0.1', '300'],
                             meta={'b': 1}),]
    assert ref_bonds == test_molecule.interactions['bonds']


@pytest.mark.parametrize('atoms, bonds, interactions, removed, expected', [
    # empty molecule
    ([], [], [], {}, {}),
    # interactions that all need to be removed
    ([1, 2], [(1, 2)], [('bond', (1, 2), {})], {'type_': 'bond', "atoms": (1, 2)}, {}),
    # Molecule with interactions of different types of which some need to be removed
    (
        [1, 2, 3, 4, 5, 6, 7],
        [(1, 2), (3, 4), (5, 7), (6, 7)],
        [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})],
        {'type_': 'bond', 'atoms': (5, 4)},
        {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})],
         'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={})]}
    ),
    (
        [1, 2, 3, 4],
        [(1, 2), (2, 3), (3, 4)],
        [('bond', (1, 2), {}), ('bond', (2, 3), {}), ('bond', (2, 3), {}, {'version': 1})],
        {'type_': 'bond', 'atoms': (2, 3), 'version': 1},
        {'bond': [
            vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={}),
            vermouth.molecule.Interaction(atoms=(2, 3), meta={}, parameters={}),
        ]},
    ),
    (
            [1, 2, 3, 4, 5, 6, 7],
            [(1, 2), (3, 4), (5, 7), (6, 7)],
            [('bond', (1, 6), [1, 2, 3], {'comment': 'a'}), ('bond', (5, 4), {'comment': 'b'}), ('angle', (1, 2), (4, 5, 6), {'version': 1})],
            {'type_': 'bond', 'atoms': (5, 4)},
            {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={'version': 1}, parameters=(4, 5, 6))],
             'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={'comment': 'a'}, parameters=[1, 2, 3])]}
    ),

])
def test_remove_interaction(atoms, bonds, interactions, removed, expected):
    """
    Test whether molecule.remove_node also removes the corresponding
    interactions
    """
    molecule = vermouth.molecule.Molecule()
    molecule.add_nodes_from(atoms)
    molecule.add_edges_from(bonds)
    for type_, atoms, params, *meta in interactions:
        if meta:
            meta = meta[0]
        else:
            meta = None
        molecule.add_interaction(type_, atoms, params, meta=meta)

    if removed:
        molecule.remove_interaction(**removed)

    assert molecule.interactions == expected