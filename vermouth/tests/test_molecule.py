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
    return molecule.copy(as_view=False)


@pytest.fixture
def molecule_subgraph(molecule):
    return molecule.subgraph([2, 0])


@pytest.mark.xfail(reason='issue #61')
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


@pytest.mark.xfail(reason='issue #61')
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

    molecule_copy.add_interaction(
        type_='angles',
        atoms=(0, 2, 3),
        parameters=['5', '6'],
        meta={'unmutable': 2},
    )
    assert 'angles' not in molecule.interactions


@pytest.mark.xfail(reason='issue #60')
def test_subgraph_base(molecule_subgraph):
    assert tuple(molecule_subgraph) == (2, 0)  # order matters!
    assert (0, 2) in molecule_subgraph.edges
    assert (0, 1) not in molecule_subgraph.edges  # node 1 is not there


@pytest.mark.xfail(reason='issue #61')
def test_subgraph_interactions(molecule_subgraph):
    bond_atoms = [bond.atoms for bond in molecule_subgraph.interactions['bonds']]
    assert (0, 2) in bond_atoms
    assert (0, 1) not in bond_atoms


def test_link_predicate_match():
    lp = vermouth.molecule.LinkPredicate(None)
    with pytest.raises(NotImplementedError):
        lp.match(1, 2)
