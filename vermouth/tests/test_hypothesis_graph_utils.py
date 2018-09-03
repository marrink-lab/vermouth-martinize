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

from time import perf_counter

import hypothesis.strategies as st
from hypothesis import given, note, settings

from hypothesis_networkx import graph_builder
import vermouth

attrnames = ['attr1', 'attr2']
node_data = st.dictionaries(keys=st.sampled_from(attrnames), values=st.integers(min_value=-10, max_value=10))

attrs = st.lists(st.sampled_from(attrnames), unique=True, min_size=0, max_size=2)

builder = graph_builder(node_data=node_data, min_nodes=0, max_nodes=25, node_keys=st.integers(max_value=10, min_value=0))


@settings(max_examples=500)
@given(graph1=builder, graph2=builder, attrs=attrs)
def test_maximum_common_subgraph(graph1, graph2, attrs):
    expected = vermouth.graph_utils.categorical_maximum_common_subgraph(graph1, graph2, attrs)
    
    found = vermouth.graph_utils.maximum_common_subgraph(graph1, graph2, attrs)
    
    note(attrs)
    note(graph1.nodes(data=True))
    note(graph1.edges)
    note(graph2.nodes(data=True))
    note(graph2.edges)
    assert set(frozenset(f.items()) for f in found) == set(frozenset(e.items()) for e in expected)
