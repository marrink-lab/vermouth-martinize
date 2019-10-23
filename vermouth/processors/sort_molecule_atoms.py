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
Provides a processor that sorts atoms within molecules.
"""

import copy
from functools import partial
from .processor import Processor


class SortMoleculeAtoms(Processor):
    """
    Sort the atoms within a molecule by chain, resid, and resname.

    This is usefull, for instance, when atoms have been added (*e.g.* missing
    atoms identified by :class:`vermouth.processors.repair_graph.RepairGraph`).
    The atom keys are left identical, only the order of the nodes is changed.
    """
    def run_molecule(self, molecule):
        node_order = sorted(molecule, key=partial(_keyfunc, molecule))
        sorted_nodes = [
            (node_key, molecule.nodes[node_key])
            for node_key in node_order
        ]
        edges = tuple(molecule.edges(data=True))
        interactions = copy.copy(molecule.interactions)
        molecule.remove_nodes_from(node_order)
        molecule.add_nodes_from(sorted_nodes)
        molecule.add_edges_from(edges)
        molecule.interactions = interactions
        return molecule


def _keyfunc(graph, node_idx):
    """
    Reduce a molecule node to a tuple of chain, resid, and resname.
    """
    # TODO add something like idx_in_residue
    return (
        graph.nodes[node_idx]['chain'],
        graph.nodes[node_idx]['resid'],
        graph.nodes[node_idx]['resname'],
    )
