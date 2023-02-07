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

from functools import partial
from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger
LOGGER = StyleAdapter(get_logger(__name__))


class SortMoleculeAtoms(Processor):
    """
    Sort the atoms within a molecule by the attributes listed in the
    :attr:`sortby_attrs`. Optionally, new atom indices are assigned to the node
    attribute :attr:`target_attr`.

    Sorting nodes is useful because a lot of software assumes chains and
    residues are listed contiguously. In particular this gets important when we
    add atoms --- for instance missing atoms identified by
    :class:`vermouth.processors.repair_graph.RepairGraph`).

    Nodes in the molecule are reordered according to the node attributes listed
    in :attr:`sortby_attrs`. The atom keys are left identical, only the order
    of the nodes is changed. Optionally, the new indices can be assigned to
    nodes :attr:`target_attr` attribute.

    Attributes
    ----------
    sortby_attrs: collections.abc.Sequence[collections.abc.Hashable]
        Nodes will be sorted by these node attributes.
    target_attr: collections.abc.Hashable
        If not ``None``, new indices will be assigned to this node attribute,
        starting with 1. It is a good idea to make sure this attribute is also
        listed in :attr:`sortby_attrs` so that the sorting is stable.
    """
    def __init__(self, sortby_attrs=('chain', 'resid', 'resname', 'insertion_code', 'atomid'),
                 target_attr=None):
        self.sortby_attrs = sortby_attrs
        self.target_attr = target_attr
        if self.target_attr is not None and  self.target_attr not in self.sortby_attrs:
            LOGGER.warning("{} is not in {}. Atom sorting may be unstable.",
                           self.target_attr, self.sortby_attrs)

    def run_molecule(self, molecule):
        node_order = sorted(molecule, key=partial(_keyfunc, molecule, attrs=self.sortby_attrs))
        for new_idx, node_key in enumerate(node_order, 1):
            molecule._node.move_to_end(node_key)
            if self.target_attr is not None:
                molecule.nodes[node_key][self.target_attr] = new_idx
        return molecule


def _keyfunc(graph, node_idx, attrs):
    """
    Reduce a molecule node to a tuple of chain, resid, and resname.
    """
    return [graph.nodes[node_idx].get(attr) for attr in attrs]
