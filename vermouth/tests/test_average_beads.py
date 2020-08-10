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
Tests for the :mod:`vermouth.processors.average_beads` module.
"""

import pytest
import networkx as nx
import numpy as np

from vermouth.processors import average_beads

# pylint: disable=redefined-outer-name

@pytest.fixture
def mol_with_subgraph():
    """
    Create a molecule with a subgraph under the "graph" attribute of each node.
    """
    mol = nx.OrderedGraph()

    subgraph = nx.Graph()
    subgraph.add_nodes_from((
        (0, {'mass': 1.1, 'not mass': 2.1, 'position': np.array([1, 2, 3], dtype=float),}),
        (1, {'mass': 1.2, 'not mass': 2.2, 'position': np.array([2, 3, 4], dtype=float),}),
        (2, {'mass': 1.3, 'not mass': 2.3, 'position': np.array([3, 4, 5], dtype=float),}),
    ))
    mol.add_node(0, **{
        "graph": subgraph,
        "mapping_weights": {0: 1, 1: 2, 2: 3},
        "target mass": np.array([2.378378378, 3.378378378, 4.378378378]),
        "target not mass": np.array([2.358208955, 3.358208955, 4.358208955]),
        "target None": np.array([2.333333333, 3.333333333, 4.333333333]),
        "target False": np.array([2.333333333, 3.333333333, 4.333333333]),
    })


    subgraph = nx.Graph()
    subgraph.add_nodes_from((
        (0, {'mass': 1.2, 'not mass': 2.2, 'position': np.array([2, 3, 4], dtype=float),}),
        (1, {'mass': 1.3, 'not mass': 2.3, 'position': np.array([3, 4, 5], dtype=float),}),
    ))
    mol.add_node(1, **{
        "graph": subgraph,
        "mapping_weights": {0: 2, 1: 3},
        "target mass": np.array([2.619047619, 3.619047619, 4.619047619]),
        "target not mass": np.array([2.610619469, 3.610619469, 4.610619469]),
        "target None": np.array([2.6, 3.6, 4.6]),
        "target False": np.array([2.6, 3.6, 4.6]),
    })

    subgraph = nx.Graph()
    mol.add_node(2, **{
        "graph": subgraph,
        "mapping_weights": {},
        "target mass": np.array([np.nan, np.nan, np.nan]),
        "target not mass": np.array([np.nan, np.nan, np.nan]),
        "target None": np.array([np.nan, np.nan, np.nan]),
        "target False": np.array([np.nan, np.nan, np.nan]),
    })

    subgraph = nx.Graph()
    subgraph.add_nodes_from((
        (0, {'mass': 1.2, 'not mass': 2.2, 'position': np.array([2, 3, 4], dtype=float),}),
        (1, {'mass': 1.3, 'not mass': 2.3, 'position': np.array([3, 4, 5], dtype=float),}),
    ))
    mol.add_node(3, **{
        "graph": subgraph,
        "mapping_weights": {0: 0, 1: 0},
        "target mass": np.array([np.nan, np.nan, np.nan]),
        "target not mass": np.array([np.nan, np.nan, np.nan]),
        "target None": np.array([np.nan, np.nan, np.nan]),
        "target False": np.array([np.nan, np.nan, np.nan]),
    })

    return mol

@pytest.fixture(params=(None, 'mass', 'not mass'))
def mol_with_variable(request, mol_with_subgraph):
    """
    Build a mock molecule with a mock force field declaring 'center_weight'.
    """
    weight = request.param

    class MockForceField:
        pass

    ff = MockForceField()
    ff.variables = {'center_weight': weight}

    mol_with_subgraph.force_field = ff
    return mol_with_subgraph


@pytest.mark.parametrize('weight', (None, 'mass', 'not mass'))
def test_do_average_bead(mol_with_subgraph, weight):
    """
    Test normal operation of :func:`average_beads.do_average_bead`.
    """
    result_mol = average_beads.do_average_bead(
        mol_with_subgraph, ignore_missing_graphs=False, weight=weight,
    )
    target_key = 'target {}'.format(weight)
    target_positions = np.stack([node[target_key] for node in mol_with_subgraph.nodes.values()])
    positions = np.stack([node['position'] for node in mol_with_subgraph.nodes.values()])
    assert np.allclose(positions, target_positions, equal_nan=True)


@pytest.mark.parametrize('weight', ('mass', 'not mass'))
def test_shoot_weight(mol_with_subgraph, weight):
    """
    Test that :func:`average_beads.do_average_bead` fails if a weight is missing.
    """
    del mol_with_subgraph.nodes[0]['graph'].nodes[1][weight]
    with pytest.raises(KeyError):
        average_beads.do_average_bead(
            mol_with_subgraph, ignore_missing_graphs=False, weight=weight,
        )


def test_shoot_graph(mol_with_subgraph):
    """
    Test that :func:`average_beads.do_average_bead` fails if a subgraph is missing.
    """
    del mol_with_subgraph.nodes[1]['graph']
    with pytest.raises(ValueError):
        average_beads.do_average_bead(mol_with_subgraph)


def test_processor_variable(mol_with_variable):
    processor = average_beads.DoAverageBead()
    mol = processor.run_molecule(mol_with_variable)
    weight = mol_with_variable.force_field.variables['center_weight']
    target_key = 'target {}'.format(weight)
    target_positions = np.stack([node[target_key] for node in mol_with_variable.nodes.values()])
    positions = np.stack([node['position'] for node in mol_with_variable.nodes.values()])
    assert np.allclose(positions, target_positions, equal_nan=True)


@pytest.mark.parametrize('weight', (False, 'mass', 'not mass'))
def test_processor_weight(mol_with_variable, weight):
    processor = average_beads.DoAverageBead(weight=weight)
    mol = processor.run_molecule(mol_with_variable)
    target_key = 'target {}'.format(weight)
    target_positions = np.stack([node[target_key] for node in mol_with_variable.nodes.values()])
    positions = np.stack([node['position'] for node in mol_with_variable.nodes.values()])
    assert np.allclose(positions, target_positions, equal_nan=True)
