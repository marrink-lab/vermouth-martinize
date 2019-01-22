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
Hypothesis based unittests for ``vermouth.graph_utils``
"""
import networkx as nx

import hypothesis.strategies as st
from hypothesis import event, given, note, settings

from hypothesis_networkx import graph_builder
import vermouth

from .helper_functions import make_into_set


# no-value-for-parameter because `draw` is not explicitely passed;
# no-member because module networkx does indeed have a member isomorphism.
# pylint: disable=no-value-for-parameter, no-member


MAX_NODES = 10
ATTRNAMES = ['attr1', 'attr2']
NODE_DATA = st.dictionaries(keys=st.sampled_from(ATTRNAMES),
                            values=st.integers(min_value=0, max_value=MAX_NODES))

ATTRS = st.lists(st.sampled_from(ATTRNAMES), unique=True, min_size=0, max_size=2)

MCS_BUILDER = graph_builder(node_data=NODE_DATA, min_nodes=0, max_nodes=MAX_NODES,
                            node_keys=st.integers(max_value=MAX_NODES, min_value=0))


@settings(max_examples=500, deadline=None)
@given(graph1=MCS_BUILDER, graph2=MCS_BUILDER, attrs=ATTRS)
def test_maximum_common_subgraph(graph1, graph2, attrs):
    """
    Test ``maximum_common_subgraph`` against
    ``categorical_maximum_common_subgraph`` as reference implementation.
    """
    expected = vermouth.graph_utils.categorical_maximum_common_subgraph(graph1, graph2, attrs)

    found = vermouth.graph_utils.maximum_common_subgraph(graph1, graph2, attrs)

    note(("Attributes that must match", attrs))
    note(("Graph 1 nodes", graph1.nodes(data=True)))
    note(("Graph 1 edges", graph1.edges))
    note(("Graph 2 nodes", graph2.nodes(data=True)))
    note(("Graph 2 edges", graph2.edges))
    # We don't find all MCS'es. See comment in
    # vermouth.graph_utils.maximum_common_subgraph
    found = make_into_set(found)
    expected = make_into_set(expected)

    if found == expected:
        event("Exact match")
    assert found <= expected


ISO_DATA = st.fixed_dictionaries({'atomname': st.integers(max_value=MAX_NODES, min_value=0),
                                  'element': st.integers(max_value=MAX_NODES, min_value=0)})

ISO_BUILDER = graph_builder(node_data=ISO_DATA, min_nodes=0, max_nodes=MAX_NODES,
                            node_keys=st.integers(max_value=MAX_NODES, min_value=0))


@settings(max_examples=500)
@given(reference=ISO_BUILDER, graph=ISO_BUILDER)
def test_isomorphism_nonmatch(reference, graph):
    """
    Test ``isomorphism`` against ``networkx.isomorphism.GraphMatcher.subgraph_isomorphisms_iter``
    as reference implementation using graph that are probably not subgraphs.
    """

    note(("Reference nodes", reference.nodes(data=True)))
    note(("Reference edges", reference.edges))
    note(("Graph nodes", graph.nodes(data=True)))
    note(("Graph edges", graph.edges))

    node_match = nx.isomorphism.categorical_node_match('element', None)
    matcher = nx.isomorphism.GraphMatcher(reference, graph, node_match=node_match)
    expected = make_into_set(matcher.subgraph_isomorphisms_iter())
    found = make_into_set(vermouth.graph_utils.isomorphism(reference, graph))
    note(("Found", found))
    note(("Expected", expected))

    if not expected:
        event("Not subgraphs")
    if found == expected:
        event("Exact match")

    assert found <= expected


@settings(max_examples=500)
@given(st.data())
def test_isomorphism_match(data):
    """
    Test ``isomorphism`` against ``networkx.isomorphism.GraphMatcher.subgraph_isomorphisms_iter``
    as reference implementation using graph that are subgraphs.
    """

    reference = data.draw(ISO_BUILDER)
    nodes = data.draw(st.sets(st.sampled_from(list(reference.nodes)),
                              max_size=len(reference)))
    graph = reference.subgraph(nodes)

    note(("Reference nodes", reference.nodes(data=True)))
    note(("Reference edges", reference.edges))
    note(("Graph nodes", graph.nodes(data=True)))
    note(("Graph edges", graph.edges))

    node_match = nx.isomorphism.categorical_node_match('element', None)
    matcher = nx.isomorphism.GraphMatcher(reference, graph, node_match=node_match)
    expected = make_into_set(matcher.subgraph_isomorphisms_iter())
    found = make_into_set(vermouth.graph_utils.isomorphism(reference, graph))

    note(("Found", found))
    note(("Expected", expected))

    if not expected:
        event("Not subgraphs")
    if found == expected:
        event("Exact match")

    assert found <= expected
