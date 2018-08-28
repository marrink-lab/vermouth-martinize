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

import networkx as nx

import hypothesis.strategies as st
from hypothesis import given, note, settings

from hypothesis_networkx import graph_builder
import vermouth


max_nodes = 10
attrnames = ['attr1', 'attr2']
node_data = st.dictionaries(keys=st.sampled_from(attrnames), values=st.integers(min_value=0, max_value=max_nodes))

attrs = st.lists(st.sampled_from(attrnames), unique=True, min_size=0, max_size=2)

MCS_builder = graph_builder(node_data=node_data, min_nodes=0, max_nodes=max_nodes, node_keys=st.integers(max_value=max_nodes, min_value=0))


@settings(max_examples=500)
@given(graph1=MCS_builder, graph2=MCS_builder, attrs=attrs)
def test_maximum_common_subgraph(graph1, graph2, attrs):
    expected = vermouth.graph_utils.categorical_maximum_common_subgraph(graph1, graph2, attrs)
    
    found = vermouth.graph_utils.maximum_common_subgraph(graph1, graph2, attrs)
    
    note(attrs)
    note(graph1.nodes(data=True))
    note(graph1.edges)
    note(graph2.nodes(data=True))
    note(graph2.edges)
    # We don't find all MCS'es. See comment in vermouth.graph_utils.maximum_common_subgraph
    assert set(frozenset(f.items()) for f in found) <= set(frozenset(e.items()) for e in expected)


iso_data = st.fixed_dictionaries({'atomname': st.integers(max_value=max_nodes, min_value=0),
                                  'element': st.integers(max_value=max_nodes, min_value=0)})
iso_builder = graph_builder(node_data=iso_data, min_nodes=0, max_nodes=max_nodes, node_keys=st.integers(max_value=max_nodes, min_value=0))

@settings(max_examples=500)
@given(reference=iso_builder, graph=iso_builder)
def test_isomorphism_nonmatch(reference, graph):
    
    note(reference.nodes(data=True))
    note(reference.edges)
    note(graph.nodes(data=True))
    note(graph.edges)
    
    matcher = nx.isomorphism.GraphMatcher(reference, graph, node_match=nx.isomorphism.categorical_node_match('element', None))
    expected = set(frozenset(match.items()) for match in matcher.subgraph_isomorphisms_iter())
    found = list(vermouth.graph_utils.isomorphism(reference, graph))
    found = set(frozenset(match.items()) for match in found)
    note(found)
    note(expected)

    assert found <= expected


@settings(max_examples=500)
@given(st.data())
def test_isomorphism_match(data):
    
    reference = data.draw(iso_builder)
    nodes = data.draw(st.sets(st.sampled_from(list(reference.nodes)), max_size=len(reference)))
    graph = reference.subgraph(nodes)
    
    note(reference.nodes(data=True))
    note(reference.edges)
    note(graph.nodes(data=True))
    note(graph.edges)
    
    matcher = nx.isomorphism.GraphMatcher(reference, graph, node_match=nx.isomorphism.categorical_node_match('element', None))
    expected = set(frozenset(match.items()) for match in matcher.subgraph_isomorphisms_iter())
    found = list(vermouth.graph_utils.isomorphism(reference, graph))
    
    found = set(frozenset(match.items()) for match in found)
    note(found)
    note(expected)
    assert found <= expected
