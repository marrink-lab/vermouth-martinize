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
Contains unittests for the functions in vermouth.graph_utils.
"""


from pprint import pprint

import networkx as nx
import pytest
import vermouth
from .helper_functions import make_into_set

# no-member because module networkx does indeed have a member isomorphism;
# too-many-arguments because some tests just need a lot of data.
# pylint: disable=no-member, too-many-arguments


def basic_molecule(node_data, edge_data=None):
    """
    Construct a simple Molecule based with specified nodes and edges.
    """
    if edge_data is None:
        edge_data = {}

    mol = vermouth.Molecule()
    for idx, node in enumerate(node_data):
        mol.add_node(idx, **node)
    for (idx, jdx), data in edge_data.items():
        mol.add_edge(idx, jdx, **data)
    return mol


@pytest.mark.parametrize('node_data_in, expected_node_data', [
    ([], []),
    ([{'atomname': 'H3'}],
     [{'atomname': 'H3', 'element': 'H'}]),
    ([{'atomname': '1H3'}],
     [{'atomname': '1H3', 'element': 'H'}]),
    ([{'atomname': 'H3'}, {'atomname': '1H3'}],
     [{'atomname': 'H3', 'element': 'H'}, {'atomname': '1H3', 'element': 'H'}]),
    ([{'atomname': 'Cl1', 'element': 'Cl', 'attr': None},
      {'atomname': '31C3'}],
     [{'atomname': 'Cl1', 'element': 'Cl', 'attr': None},
      {'atomname': '31C3', 'element': 'C'}]),
    ([{'element': 'Cl'}, {'atomname': '31C3'}],
     [{'element': 'Cl'}, {'atomname': '31C3', 'element': 'C'}]),
])
def test_add_element_attr(node_data_in, expected_node_data):
    """
    Tests for the function ``add_element_attr``.
    """
    mol = basic_molecule(node_data_in)
    vermouth.graph_utils.add_element_attr(mol)
    expected = basic_molecule(expected_node_data)
    assert mol.nodes(data=True) == expected.nodes(data=True)


@pytest.mark.parametrize('node_data_in, exception', [
    ([{'atomname': '1234'}], ValueError),  # No alpha character
    ([{'peanuts': '1H3'}], ValueError),  # Not atomname attribute
    ([{'atomname': 'H3'}, {'atomname': '1234'}], ValueError),  # No alpha character for second atom
    ([{'atomname': 'H3'}, {'peanuts': '1234'}], ValueError),  # No atomname attribute for second atom
])
def test_add_element_attr_errors(node_data_in, exception):
    """
    Tests to make sure the function ``add_element_attr`` raises the expected
    errors.
    """
    mol = basic_molecule(node_data_in)
    with pytest.raises(exception):
        vermouth.graph_utils.add_element_attr(mol)


@pytest.mark.parametrize('node_data1, node_data2, attrs, expected', [
    ([], [], [], {}),
    ([{}], [{}], [], {(0, 0): {}}),
    (
        [{'a': 1}],
        [{'a': 2}],
        [],
        {(0, 0): {'a': (1, 2)}}
    ),
    (
        [{'a': 1}],
        [{'a': 1}],
        ['a'],
        {(0, 0): {'a': (1, 1)}}
    ),
    (
        [{'a': 1}],
        [{'a': 2}],
        ['a'],
        {}
    ),
    (
        [{'a': 1}],
        [{'b': 1}],
        ['a'],
        {}
    ),
    (
        [{'a': 1, 'b': 2}],
        [{'a': 1, 'b': 1}],
        ['a'],
        {(0, 0): {'a': (1, 1), 'b': (2, 1)}}
    ),
    (
        [{"1": 1, "2": 2}],
        [{"2": 2, "3": 3}],
        ["2"],
        {(0, 0): {"1": (1, None), "2": (2, 2), "3": (None, 3)}}
    ),
    (
        [{'a': 1, 'b': 2}],
        [{'a': 1, 'b': 1}],
        ['a', 'b'],
        {}
    ),
    (
        [{'a': 1}, {'b': 2}],
        [{'b': 2}, {'c': 3}],
        ['b'],
        {(1, 0): {'b': (2, 2)}, (0, 1): {'a': (1, None), 'c': (None, 3)}}
    ),
    (
        [{'a': 1, 'b': 1}, {'b': 2}],
        [{'b': 2}, {'c': 3, 'b': 2}],
        ['b'],
        {(1, 0): {'b': (2, 2)}, (1, 1): {'b': (2, 2), 'c': (None, 3)}}
    ),
    (
        [{'a': 1, 'b': 1}, {'b': 2, 'a': 2}],
        [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}],
        ['a'],
        {(0, 0): {'a': (1, 1), 'b': (1, 1)}, (1, 1): {'a': (2, 2), 'b': (2, 2)}}
    ),
])
def test_categorical_cartesian_product(node_data1, node_data2, attrs, expected):
    """
    Tests for the function ``categorical_cartesian_product``.
    """
    graph1 = basic_molecule(node_data1)
    graph2 = basic_molecule(node_data2)
    found = vermouth.graph_utils.categorical_cartesian_product(graph1, graph2, attrs)
    expected_mol = nx.Graph()

    # Only test nodes, because the categorical product does not contain edges.
    for idx, data in expected.items():
        expected_mol.add_node(idx, **data)
    assert expected_mol.nodes(data=True) == found.nodes(data=True)


@pytest.mark.parametrize('node_data1, edges1, node_data2, edges2, attrs, expected_nodes, expected_edges', [
    ([], {}, [], {}, [], {}, {}),
    (
        [{}, {}], {},
        [{}, {}], {(0, 1): {}},
        [],
        {(0, 0): {}, (1, 0): {}, (0, 1): {}, (1, 1): {}},
        {}
    ),
    (
        [{'a': 1}, {'a': 2}], {},
        [{'a': 1}, {'a': 2}], {},
        ['a'],
        {(0, 0): {'a': (1, 1)}, (1, 1): {'a': (2, 2)}},
        {((0, 0), (1, 1)): {}}
    ),
    (
        [{}, {}], {(0, 1): {}},
        [{}, {}], {(0, 1): {}},
        [],
        {(0, 0): {}, (1, 0): {}, (0, 1): {}, (1, 1): {}},
        {((0, 0), (1, 1)): {}, ((0, 1), (1, 0)): {}}
    ),
    (
        [{}, {}], {},
        [{}, {}], {},
        [],
        {(0, 0): {}, (1, 0): {}, (0, 1): {}, (1, 1): {}},
        {((0, 0), (1, 1)): {}, ((0, 1), (1, 0)): {}}
    ),
    (
        [{}, {}], {(0, 1): {'a': 1}},
        [{}, {}], {(0, 1): {'b': 1}},
        [],
        {(0, 0): {}, (1, 0): {}, (0, 1): {}, (1, 1): {}},
        {((0, 0), (1, 1)): {'a': (1, None), 'b': (None, 1)},
         ((0, 1), (1, 0)): {'a': (1, None), 'b': (None, 1)}}
    ),
])
def test_categorical_modular_product(node_data1, edges1, node_data2, edges2,
                                     attrs, expected_nodes, expected_edges):
    """
    Tests for the function ``categorical_modular_product``.
    """
    graph1 = basic_molecule(node_data1, edges1)
    graph2 = basic_molecule(node_data2, edges2)
    found = vermouth.graph_utils.categorical_modular_product(graph1, graph2, attrs)
    expected_mol = nx.Graph()
    for idx, data in expected_nodes.items():
        expected_mol.add_node(idx, **data)
    for edge_idx, data in expected_edges.items():
        expected_mol.add_edge(*edge_idx, **data)

    assert found.nodes(data=True) == expected_mol.nodes(data=True)
    edges_seen = set()
    for idx, jdx, data in found.edges(data=True):
        assert expected_mol.edges[idx, jdx] == data
        edges_seen.add(frozenset((idx, jdx)))
    assert set(frozenset(edge) for edge in expected_mol.edges) == edges_seen


@pytest.mark.parametrize('node_data1, edges1, node_data2, edges2, attrs, expected', [
    ([], {}, [], {}, ['id'], []),
    (
        [{'id': 1}], {},
        [{'id': 1}], {},
        ['id'],
        [{0: 0}]
    ),
    (
        [{'id': 1}, {'id': 2}], {},
        [{'id': 1}, {'id': 2}], {},
        ['id'],
        [{0: 0, 1: 1}]
    ),
    (
        [{'id': 1}, {'id': 2}], {(0, 1): {}},
        [{'id': 1}, {'id': 2}], {},
        ['id'],
        [{0: 0}, {1: 1}]
    ),
    (
        [{'id': 0}, {'id': 1}, {'id': 2}], {(0, 1): {}, (1, 2): {}},
        [{'id': 1}, {'id': 2}, {'id': 3}], {(0, 1): {}, (1, 2): {}},
        ['id'],
        [{1: 0, 2: 1}]
    ),
    (
        [{}, {}], {(0, 1): {}},
        [{}, {}], {(0, 1): {}},
        [],
        [{0: 0, 1: 1}, {0: 1, 1: 0}]
    ),
])
def test_categorical_maximum_common_subgraph(node_data1, edges1, node_data2,
                                             edges2, attrs, expected):
    """
    Tests for the function ``categorical_maximum_common_subgraph``.
    """
    graph1 = basic_molecule(node_data1, edges1)
    graph2 = basic_molecule(node_data2, edges2)
    found = vermouth.graph_utils.categorical_maximum_common_subgraph(graph1, graph2, attrs)
    assert make_into_set(found) == make_into_set(expected)


@pytest.mark.parametrize('node_data, edges, partitions, attrs, expected_nodes, expected_edges', [
    ([], {}, [], {}, [], {}),
    ([{}], {}, [[0]], {}, [{}], {}),
    (
        [{}, {}],
        {},
        [[0, 1]],
        {},
        [{}],
        {}
    ),
    (
        [{}, {}],
        {(0, 1): {}},
        [[0, 1]],
        {},
        [{}],
        {}
    ),
    (
        [{}, {}],
        {(0, 1): {}},
        [[0, 1]],
        {'id': [0]},
        [{'id': 0}],
        {}
    ),
    (
        [{}, {}, {}, {}],
        {(0, 1): {}, (1, 2): {}, (2, 3): {}},
        [[0, 1], [2, 3]],
        {'id': [0, 1]},
        [{'id': 0}, {'id': 1}],
        {(0, 1): {'weight': 1.0}}
    ),
    (
        [{}, {}, {}, {}],
        {(0, 1): {}, (1, 2): {'weight': 0.5}, (2, 3): {}},
        [[0, 1], [2, 3]],
        {'id': [0, 1]},
        [{'id': 0}, {'id': 1}],
        {(0, 1): {'weight': 0.5}}
    ),
    (
        [{}, {}, {}, {}],
        {(0, 1): {}, (2, 3): {}},
        [[0, 1], [2, 3]],
        {'id': [0, 1]},
        [{'id': 0}, {'id': 1}],
        {}
    ),
    (
        [{}, {}, {}, {}],
        {(0, 1): {}, (2, 3): {}},
        [[0, 1], [2]],
        {'id': [0, 1]},
        [{'id': 0}, {'id': 1}],
        {}
    ),
    (
        [{}, {}, {}, {}],
        {(0, 1): {}, (2, 3): {}},
        [[0, 1]],
        {'id': [0]},
        [{'id': 0}],
        {}
    ),
    (
        [{}, {}, {}, {}],
        {(0, 1): {}, (1, 2): {}, (2, 3): {}, (3, 0): {'weight': 0.5}},
        [[0, 1], [2, 3]],
        {'id': [0, 1]},
        [{'id': 0}, {'id': 1}],
        {(0, 1): {'weight': 1.5}}
    ),
    (
        [{}, {}, {}, {}],
        {(0, 1): {}, (1, 2): {}, (2, 3): {}},
        [[0, 1], [2, 3]],
        {'id': [0, 1], 'attr': ['a', 'b']},
        [{'id': 0, 'attr': 'a'}, {'id': 1, 'attr': 'b'}],
        {(0, 1): {'weight': 1.0}}
    ),
    pytest.param([{}, {}, {}, {}],
                 {(0, 1): {}, (1, 2): {}, (2, 3): {}},
                 [[0, 1], [2, 3]],
                 {'id': [1], 'attr': ['a', 'b']},
                 [{'id': 0, 'attr': 'a'}, {'id': 1, 'attr': 'b'}],
                 {(0, 1): {'weight': 1.0}},
                 marks=pytest.mark.xfail(raises=IndexError)),
])
def test_blockmodel(node_data, edges, partitions, attrs, expected_nodes, expected_edges):
    """
    Tests for the function ``blockmodel``.
    """
    graph = basic_molecule(node_data, edges)
    found = vermouth.graph_utils.blockmodel(graph, partitions, **attrs)
    expected = basic_molecule(expected_nodes, expected_edges)
    pprint(("Found nodes", found.nodes(data=True)))
    pprint(("Expected nodes", expected.nodes(data=True)))

    for node in found:
        data = found.nodes[node]
        subgraph = data['graph']
        assert len(subgraph.nodes) == data['nnodes']
        assert len(subgraph.edges) == data['nedges']
        assert nx.density(subgraph) == data['density']
        del found.nodes[node]['graph']
        del found.nodes[node]['nnodes']
        del found.nodes[node]['nedges']
        del found.nodes[node]['density']

    assert found.nodes(data=True) == expected.nodes(data=True)
    edges_seen = set()
    for idx, jdx, data in found.edges(data=True):
        assert expected.has_edge(idx, jdx) and expected.edges[idx, jdx] == data
        edges_seen.add(frozenset((idx, jdx)))
    assert set(frozenset(edge) for edge in expected.edges) == edges_seen


def test_blockmodel_graph_attr():
    """
    Make sure the function ``blockmodel`` produces node attributes ``'graph'``,
    ``'nnodes'`` and ``'nedges'`` that have the correct values.
    """
    graph = basic_molecule([{}, {}, {}], {(0, 1): {}, (1, 2): {}})
    found = vermouth.graph_utils.blockmodel(graph, [[0], [1, 2]])
    assert found.nodes[0]['nnodes'] == 1
    assert found.nodes[0]['nnodes'] == len(found.nodes[0]['graph'].nodes)
    assert found.nodes[1]['nnodes'] == 2
    assert found.nodes[1]['nnodes'] == len(found.nodes[1]['graph'].nodes)
    assert found.nodes[0]['nedges'] == 0
    assert found.nodes[0]['nedges'] == len(found.nodes[0]['graph'].edges)
    assert found.nodes[1]['nedges'] == 1
    assert found.nodes[1]['nedges'] == len(found.nodes[1]['graph'].edges)


@pytest.mark.parametrize('nodes1, nodes2, match, expected', [
    ([], [], {}, 0),
    (
        [{'atomname': 0}],
        [{'atomname': 1}],
        {0: 0}, 0
    ),
    (
        [{'atomname': 0}],
        [{'atomname': 1}],
        {}, 0
    ),
    (
        [{'atomname': 0}],
        [{'atomname': 0}],
        {}, 0
    ),
    (
        [{'atomname': 0}],
        [{'atomname': 0}],
        {0: 0}, 1
    ),
    (
        [{'atomname': 0}, {'atomname': 0}],
        [{'atomname': 0}, {'atomname': 0}],
        {0: 0}, 1
    ),
    (
        [{'atomname': 0}, {'atomname': 0}],
        [{'atomname': 0}, {'atomname': 0}],
        {0: 0, 1: 1}, 2
    ),
    (
        [{'atomname': 0}, {'atomname': 0}],
        [{'atomname': 0}, {'atomname': 1}],
        {0: 0, 1: 1}, 1
    ),
    (
        [{'atomname': 0}, {'atomname': 1}],
        [{'atomname': 0}, {'atomname': 1}],
        {0: 0, 1: 1}, 2
    ),
    (
        [{'atomname': 0}, {'atomname': 1}],
        [{'atomname': 0}, {'atomname': 1}],
        {0: 1}, 0
    ),
])
def test_rate_match(nodes1, nodes2, match, expected):
    """
    Tests for the function ``rate_match``.
    """
    mol1 = basic_molecule(nodes1)
    mol2 = basic_molecule(nodes2)
    found = vermouth.graph_utils.rate_match(mol1, mol2, match)
    assert found == expected


@pytest.mark.parametrize('nodes1, edges1, nodes2, edges2', [
    ([], {}, [], {}),
    (
        [{'chain': 0, 'resid': 0, 'resname': 0}], {},
        [{'chain': 0, 'resid': 0, 'resname': 0, 'atomname': 0}], {}
    ),
    (
        [{'chain': 0, 'resid': 0, 'resname': 1}], {},
        [{'chain': 0, 'resid': 0, 'resname': 1, 'atomname': 1}], {}
    ),
    (
        [{'chain': 0, 'resid': 2, 'resname': 1}], {},
        [{'chain': 0, 'resid': 2, 'resname': 1, 'atomname': 1}], {}
    ),
    (
        [{'chain': 0, 'resid': 2, 'resname': 1, 'attr': 5}], {},
        [{'chain': 0, 'resid': 2, 'resname': 1, 'atomname': 1}], {}
    ),
    (
        [{'chain': 0, 'resid': 2, 'resname': 1, 'attr': 5},
         {'chain': 0, 'resid': 2, 'resname': 1, 'attr': 7}],
        {},
        [{'chain': 0, 'resid': 2, 'resname': 1, 'atomname': 1}],
        {}
    ),
    (
        [{'chain': 0, 'resid': 2, 'resname': 1, 'attr': 5},
         {'chain': 0, 'resid': 2, 'resname': 2, 'attr': 7}],
        {},
        [{'chain': 0, 'resid': 2, 'resname': 1, 'atomname': 1},
         {'chain': 0, 'resid': 2, 'resname': 2, 'atomname': 2}],
        {}
    ),
    (
        [{'chain': 0, 'resid': 2, 'resname': 1, 'attr': 5},
         {'chain': 0, 'resid': 2, 'resname': 2, 'attr': 7}],
        {(0, 1): {}},
        [{'chain': 0, 'resid': 2, 'resname': 1, 'atomname': 1},
         {'chain': 0, 'resid': 2, 'resname': 2, 'atomname': 2}],
        {(0, 1): {'weight': 1}}
    ),
    (
        [{'chain': 0, 'resid': 2, 'resname': 1, 'attr': 5},
         {'chain': 0, 'resid': 2, 'resname': 1, 'attr': 6},
         {'chain': 0, 'resid': 2, 'resname': 2, 'attr': 7}],
        {(2, 1): {}},
        [{'chain': 0, 'resid': 2, 'resname': 1, 'atomname': 1},
         {'chain': 0, 'resid': 2, 'resname': 2, 'atomname': 2}],
        {(0, 1): {'weight': 1}}
    ),
])
def test_make_residue_graph(nodes1, edges1, nodes2, edges2):
    """
    Tests for the function ``make_residue_graph``.
    """
    mol1 = basic_molecule(nodes1, edges1)
    found = vermouth.graph_utils.make_residue_graph(mol1)
    expected = basic_molecule(nodes2, edges2)

    for node in found:
        data = found.nodes[node]
        subgraph = data['graph']
        assert len(subgraph.nodes) == data['nnodes']
        assert len(subgraph.edges) == data['nedges']
        assert nx.density(subgraph) == data['density']
        del found.nodes[node]['graph']
        del found.nodes[node]['nnodes']
        del found.nodes[node]['nedges']
        del found.nodes[node]['density']
    pprint(("Found nodes", found.nodes(data=True)))
    pprint(("Expected nodes", expected.nodes(data=True)))

    assert found.nodes(data=True) == expected.nodes(data=True)
    edges_seen = set()
    for idx, jdx, data in found.edges(data=True):
        assert expected.has_edge(idx, jdx) and expected.edges[idx, jdx] == data
        edges_seen.add(frozenset((idx, jdx)))
    assert set(frozenset(edge) for edge in expected.edges) == edges_seen
