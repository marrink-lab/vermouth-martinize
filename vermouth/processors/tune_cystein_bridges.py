#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

def prune_edges_between_selections(molecule, selection_a, selection_b):
    """
    Remove edges which have their ends part of given selections.

    An edge is removed if has one end that is part of 'selection_a', and the
    other end part of 'selection_b'.

    Parameters
    ----------
    molecule: nx.Graph
        Molecule to prune in-place.
    selection_a, selection_b: list
        Lists of node keys from the molecule.
    """
    selection_a = set(selection_a)
    selection_b = set(selection_b)
    # to_prune is the list of edges to remove. It cannot be a ganerator or
    # removing the edges will be modifying the datastructure on which we
    # iterate.
    to_prune = [
        edge for edge in molecule.edges
        if (edge[0] in selection_a and edge[1] in selection_b)
            or (edge[1] in selection_a and edge[0] in selection_b)
    ]
    molecule.remove_edges_from(to_prune)
