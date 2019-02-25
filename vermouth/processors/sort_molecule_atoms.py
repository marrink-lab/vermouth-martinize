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
        # remove_nodes_from would be better, but it leads to an OrderedDict
        # being modified during iteration, and python does not like that.
        for node in node_order:
            super(type(molecule), molecule).remove_node(node)
        molecule.add_nodes_from(sorted_nodes)
        return molecule


def _keyfunc(graph, node_idx):
    """
    Reduce a molecule node to a tuple of chain, resid, and resname.
    """
    # TODO add something like idx_in_residue
    return (
        graph.node[node_idx]['chain'],
        graph.node[node_idx]['resid'],
        graph.node[node_idx]['resname'],
    )
