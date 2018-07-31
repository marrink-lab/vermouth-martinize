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

import itertools


def expand_isomorphism(reference, graph, isomorphism):
    """
    Since our isomorphism implementation is too smart by half, it doesn't
    return all possible isomorphisms. Instead, in the second step of the
    algorithm where it expands the isomorphisms found of non-hydrogen atoms it
    only gives the first answer for the hydrogen atoms instead of all. The
    reasoning is that those atoms are equivalent.
    Here we regenerate those cases.
    """
    isomorphism = isomorphism.copy()
    for match in isomorphism[:]:
        print('====')
        print(match)
        match = match.copy()
        for node1_1, node1_2 in itertools.combinations(match, 2):
            match1_1 = match[node1_1]
            match1_2 = match[node1_2]
            # These two isomorphisms are considered equal if their atomnames are
            # the same, they are degree 1, and they have the same neighbour.
            # First for the nodes in reference:
            degrees = reference.degree(node1_1) == 1 or reference.degree(node1_2) == 1
            names = degrees and reference.nodes[node1_1]['element'] == reference.nodes[node1_2]['element']
            neighbors = (list(reference[node1_1]) == list(reference[node1_2])) #or \
                        #(list(reference[node1_1]), [node1_1]) == ([node1_2], list(reference[node1_2]))
            ref_eligible = names and neighbors
            # Then for the nodes in graph
            degrees = graph.degree(match1_1) == 1 or graph.degree(match1_2) == 1
            names = degrees and graph.nodes[match1_1]['element'] == graph.nodes[match1_2]['element']
            neighbors = (list(graph[match1_1]) == list(graph[match1_2]))# or \
                        #(list(graph[match1_1]), [match1_1]) == ([match1_2], list(graph[match1_2]))
            graph_eligible = names and neighbors
            if graph_eligible and ref_eligible:
                # Expand the isomorphism.
                match[node1_1] = match1_2
                match[node1_2] = match1_1
                isomorphism.append(match.copy())
            print(2, match)
        isomorphism.append(match)
    return isomorphism
