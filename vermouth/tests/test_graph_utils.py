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
# limitations under the License.import pytest

import pytest
import vermouth


def basic_molecule(node_data, edge_data={}):
    mol = vermouth.Molecule()
    for idx, node in enumerate(node_data):
        mol.add_node(idx, **node)
    for (idx, jdx), data in edge_data.items():
        mol.add_edge(idx, jdx, **data)
    return mol


@pytest.mark.parametrize('node_data_in,expected_node_data', [
        ([], []),
        ([{'atomname': 'H3'}], [{'atomname': 'H3', 'element': 'H'}]),
        ([{'atomname': '1H3'}], [{'atomname': '1H3', 'element': 'H'}]),
        ([{'atomname': 'H3'}, {'atomname': '1H3'}], [{'atomname': 'H3', 'element': 'H'}, {'atomname': '1H3', 'element': 'H'}]),
        ([{'atomname': 'Cl1', 'element': 'Cl', 'attr': None}, {'atomname': '31C3'}], [{'atomname': 'Cl1', 'element': 'Cl', 'attr': None}, {'atomname': '31C3', 'element': 'C'}]),
        ([{'element': 'Cl'}, {'atomname': '31C3'}], [{'element': 'Cl'}, {'atomname': '31C3', 'element': 'C'}]),
        ]
        )
def test_add_element_attr(node_data_in, expected_node_data):
    mol = basic_molecule(node_data_in)
    vermouth.graph_utils.add_element_attr(mol)
    expected = basic_molecule(expected_node_data)
    assert mol.nodes(data=True) == expected.nodes(data=True)


@pytest.mark.parametrize('node_data_in,exception', [
        ([{'atomname': '1234'}], ValueError),
        ([{'peanuts': '1H3'}], ValueError),
        ([{'atomname': 'H3'}, {'atomname': '1234'}], ValueError),
        ([{'atomname': 'H3'}, {'peanuts': '1234'}], ValueError),
        ]
        )
def test_add_element_attr_errors(node_data_in, exception):
    mol = basic_molecule(node_data_in)
    with pytest.raises(exception):
        vermouth.graph_utils.add_element_attr(mol)


@pytest.mark.parametrize('node_data1,node_data2,attrs,expected', 
                         [
                          ([], [], [], {}),
                          ([{}], [{}], [], {(0, 0): {}}),
                          ([{'a': 1}], [{'a': 2}], [], {(0, 0): {'a': (1, 2)}}),
                          ([{'a': 1}], [{'a': 1}], ['a'], {(0, 0): {'a': (1, 1)}}),
                          ([{'a': 1}], [{'a': 2}], ['a'], {}),
                          ([{'a': 1}], [{'b': 1}], ['a'], {}),
                          ([{'a': 1, 'b': 2}], [{'a': 1, 'b': 1}], ['a'], {(0, 0): {'a': (1, 1), 'b': (2, 1)}}),
                          ([{"1": 1, "2": 2}], [{"2": 2, "3": 3}], ["2"], {(0, 0): {"1": (1, None), "2": (2, 2), "3": (None, 3)}}),
                          ([{'a': 1, 'b': 2}], [{'a': 1, 'b': 1}], ['a', 'b'], {}),
                          ([{'a': 1}, {'b': 2}], [{'b': 2}, {'c': 3}], ['b'], {(1, 0): {'b': (2, 2)}}),
                          ([{'a': 1, 'b': 1}, {'b': 2}], [{'b': 2}, {'c': 3, 'b': 2}], ['b'], {(1, 0): {'b': (2, 2)}, (1, 1): {'b': (2, 2), 'c': (None, 3)}}),
                          ([{'a': 1, 'b': 1}, {'b': 2, 'a': 2}], [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}], ['a'], {(0, 0): {'a': (1, 1), 'b': (1, 1)}, (1, 1): {'a': (2, 2), 'b': (2, 2)}}),
                         ]
                        )
def test_categorical_cartesian_product(node_data1, node_data2, attrs, expected):
    graph1 = basic_molecule(node_data1)
    graph2 = basic_molecule(node_data2)
    found = vermouth.graph_utils.categorical_cartesian_product(graph1, graph2, attrs)
    expected_mol = vermouth.Molecule()
    for idx, data in expected.items():
        expected_mol.add_node(idx, **data)
    assert expected_mol.nodes(data=True) == found.nodes(data=True)


@pytest.mark.parametrize('node_data1,edges1,node_data2,edges2,attrs,expected_nodes,expected_edges',
    [
     ([], {}, [], {}, [], {}, {}),
     ([{}, {}], {}, [{}, {}], {(0, 1): {}}, [], {(0, 0):{}, (1, 0):{}, (0, 1):{}, (1, 1):{}}, {}),
     ([{'a': 1}, {'a': 2}], {}, [{'a': 1}, {'a': 2}], {}, ['a'], {(0, 0): {'a': (1, 1)}, (1, 1): {'a': (2, 2)}}, {((0, 0), (1, 1)): {}}),
     ([{}, {}], {(0, 1): {}}, [{}, {}], {(0, 1): {}}, [], {(0, 0):{}, (1, 0):{}, (0, 1):{}, (1, 1):{}}, {((0, 0), (1, 1)): {}, ((0, 1), (1, 0)): {}}),
     ([{}, {}], {}, [{}, {}], {}, [], {(0, 0):{}, (1, 0):{}, (0, 1):{}, (1, 1):{}}, {((0, 0), (1, 1)): {}, ((0, 1), (1, 0)): {}}),
     ([{}, {}], {(0, 1): {'a': 1}}, [{}, {}], {(0, 1): {'b': 1}}, [], {(0, 0):{}, (1, 0):{}, (0, 1):{}, (1, 1):{}}, {((0, 0), (1, 1)): {'a': (1, None), 'b': (None, 1)}, ((0, 1), (1, 0)): {'a': (1, None), 'b': (None, 1)}}),
    ]
    )
def test_categorical_modular_product(node_data1, edges1, node_data2, edges2,
                                     attrs, expected_nodes, expected_edges):
    graph1 = basic_molecule(node_data1, edges1)
    graph2 = basic_molecule(node_data2, edges2)
    found = vermouth.graph_utils.categorical_modular_product(graph1, graph2, attrs)
    expected_mol = vermouth.Molecule()
    for idx, data in expected_nodes.items():
        expected_mol.add_node(idx, **data)
    for edge_idx, data in expected_edges.items():
        expected_mol.add_edge(*edge_idx, **data)

    assert found.nodes(data=True) == expected_mol.nodes(data=True)
    edges_seen = set()
    for idx, jdx, data in found.edges(data=True):
        assert expected_mol.edges[idx, jdx] == data
        edges_seen.add((idx, jdx))
    assert not expected_mol.edges - edges_seen


@pytest.mark.parametrize('node_data1,edges1,node_data2,edges2,attrs,expected',
    [
     ([], {}, [], {}, ['id'], []),
     ([{'id': 1}], {}, [{'id': 1}], {}, ['id'], [{0: 0}]),
     ([{'id': 1}, {'id': 2}], {}, [{'id': 1}, {'id': 2}], {}, ['id'], [{0: 0, 1: 1}]),
     ([{'id': 1}, {'id': 2}], {(0, 1): {}}, [{'id': 1}, {'id': 2}], {}, ['id'], [{0: 0}, {1: 1}]),
     ([{'id': 0}, {'id': 1}, {'id': 2}], {(0, 1): {}, (1, 2): {}},
      [{'id': 1}, {'id': 2}, {'id': 3}], {(0, 1): {}, (1, 2): {}}, ['id'], [{1: 0, 2: 1}]),
    ]
    )
def test_categorical_maximum_common_subgraph(node_data1, edges1, node_data2,
                                             edges2, attrs, expected):
    graph1 = basic_molecule(node_data1, edges1)
    graph2 = basic_molecule(node_data2, edges2)
    found = vermouth.graph_utils.categorical_maximum_common_subgraph(graph1, graph2, attrs)
    assert set(frozenset(f.items()) for f in found) == set(frozenset(e.items()) for e in expected)


from pprint import pprint
@pytest.mark.parametrize('node_data1,edges1,node_data2,edges2,attrs',
    [
     ([], {}, [], {}, ['id']),
     ([{'id': 1}], {}, [{'id': 1}], {}, ['id']),
     ([{'id': 1}, {'id': 2}], {}, [{'id': 1}, {'id': 2}], {}, ['id']),
     ([{'id': 1}, {'id': 2}], {(0, 1): {}}, [{'id': 1}, {'id': 2}], {}, ['id']),
     ([{'id': 1}], {}, [{'id': 1}, {'id': 1}], {(0, 1): {}}, ['id']),
     ([{'id': 1}, {'id': 1}], {(0, 1): {}}, [{'id': 1}], {}, ['id']),
     ([{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}, {'id': 2}], {(0, 1): {}, (1, 2): {}, (2, 3): {}, (3, 4): {}},
      [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}, {'id': 5}], {(0, 1): {}, (1, 2): {}, (2, 3): {}, (1, 4): {}},
      ['id']),
     ([{}, {}, {}, {}, {}], {(0, 1): {}, (0, 2): {}, (0, 4): {}, (0, 3): {}, (1, 2): {}, (1, 4): {}},
      [{}, {}, {}, {}], {(0, 2): {}, (0, 3): {}, (1, 2): {}, (1, 3): {}},
      []),
     pytest.param([{}, {}, {}, {}, {}], {(0, 1): {}, (1, 2): {}, (2, 3): {}, (2, 4): {}},
      [{}, {}, {}, {}, {}], {(0, 1): {}, (1, 2): {}, (2, 3): {}, (3, 4): {}},
      [], marks=pytest.mark.xfail),
     ]
    )
def test_maximum_common_subgraph(node_data1, edges1, node_data2, edges2, attrs):
    graph1 = basic_molecule(node_data1, edges1)
    graph2 = basic_molecule(node_data2, edges2)
    expected = vermouth.graph_utils.categorical_maximum_common_subgraph(graph1, graph2, attrs)
    found = vermouth.graph_utils.maximum_common_subgraph(graph1, graph2, attrs)
    found = set(frozenset(m.items()) for m in found)
    expected = set(frozenset(m.items()) for m in expected)
    print(len(found))
    print(len(expected))
    
    pprint(found)
    pprint(expected)
    
    pprint(expected - found)
    assert found == expected
