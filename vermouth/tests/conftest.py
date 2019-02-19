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
Configuration, setup, and hooks for pytest.
"""

from vermouth import Molecule
from vermouth.utils import are_different


def pytest_assertrepr_compare(op, left, right):
    """
    Add informative failure reports for custom types.

    See
    <https://docs.pytest.org/en/latest/assert.html#defining-your-own-explanation-for-failed-assertions>
    for more details.
    """
    if isinstance(left, Molecule) and isinstance(right, Molecule) and op == "==":
        result = ['The molecules are different.']
        if list(left.nodes()) != list(right.nodes()):
            result.append('The node keys do not match:')
            result.append('- ' + str(list(left.nodes)))
            result.append('+ ' + str(list(right.nodes)))
        else:
            for (key, left_node), (_, right_node) in zip(left.nodes(data=True), right.nodes(data=True)):
                if are_different(left_node, right_node):
                    result.append('Node {} differs:'.format(key))
                    result.append('- ' + str(left_node))
                    result.append('+ ' + str(right_node))
        unordered_left_edges = set(frozenset(edge) for edge in left.edges)
        unordered_right_edges = set(frozenset(edge) for edge in right.edges)
        if unordered_left_edges != unordered_right_edges:
            result.append('Edges are different:')
            result.append('- ' + str(unordered_left_edges))
            result.append('- ' + str(unordered_right_edges))
        else:
            for edge in left.edges:
                if are_different(left.edges[edge], right.edges[edge]):
                    result.append('The edge {} differs.'.format(edge))
                    result.append('- ' + str(left.edges[edge]))
                    result.append('+ ' + str(right.edges[edge]))
        return result
