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
import vermouth


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
    ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (3, 4), (5, 7)], [('bond', (5, 4), {}), ('bond', (1, 2), {})], [5]
     , {'bond': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})]}),

    # Molecule with interactions of which none need to be removed
    ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (3, 4)], [('bond', (5, 4), {}), ('bond', (1, 2), {})], [6, 7]
     , {'bond': [vermouth.molecule.Interaction(atoms=(5, 4), meta={}, parameters={})
                 , vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})]}),

    # Molecule with interactions of different types of which all need to be removed
    ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (3, 4), (5, 7), (6, 7)], [('bond', (1, 6), {}), ('bond', (5, 4), {})
                                                               , ('angle', (1, 2), {})], [1, 5], {}),

    # Molecule with interactions of different types of which some need to be removed
    ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (3, 4), (5, 7), (6, 7)]
     , [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})], [5]
     , {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})]
                 , 'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={})]}),

    # Molecule with interactions of different types of which none need to be removed
    ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (3, 4), (5, 7), (6, 7)]
     , [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})], [3]
     , {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})]
                 , 'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={})
                            , vermouth.molecule.Interaction(atoms=(5, 4), meta={}, parameters={})]}),
])
def test_remove_nodes_from(atoms, bonds, interactions, removed, expected):
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
    ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (3, 4), (5, 7)], [('bond', (5, 4), {}), ('bond', (1, 2), {})], 5
     , {'bond': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})]}),

    # Molecule with interactions of different types of which some need to be removed
    ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (3, 4), (5, 7), (6, 7)]
     , [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})], 5
     , {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})],
        'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={})]}),

    # Molecule with interactions of different types of which none need to be removed
    ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (3, 4), (5, 7), (6, 7)]
     , [('bond', (1, 6), {}), ('bond', (5, 4), {}), ('angle', (1, 2), {})], 3
     , {'angle': [vermouth.molecule.Interaction(atoms=(1, 2), meta={}, parameters={})],
        'bond': [vermouth.molecule.Interaction(atoms=(1, 6), meta={}, parameters={})
                 , vermouth.molecule.Interaction(atoms=(5, 4), meta={}, parameters={})]}),
])
def test_remove_node(atoms, bonds, interactions, removed, expected):
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
