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
Provides a processor that generates positions for nodes based on the weighted
average of the positions of the atoms they are constructed from.
"""


import numpy as np
from .processor import Processor


def do_average_bead(molecule, ignore_missing_graphs=False, weight=None):
    """
    Set the position of the particles to the mean of the underlying atoms.

    This requires the atoms to have a 'graph' attributes. By default, a
    :exc:`ValueError` is raised if any atom in the molecule is missing that
    'graph' attribute. This behavior can be changed by setting the
    'ignore_missing_graphs' argument to `True`, then the average positions are
    computed, but the atoms without a 'graph' attribute are skipped.

    The average is weighted using the 'mapping_weights' atom attribute. If the
    'mapping_weights' attribute is set, it has to be a dictionary with the
    atomname from the underlying graph as keys, and the weights as values.
    Atoms without a weight set use a default weight of 1.

    The average can also be weighted using an arbitrary node attribute by
    giving the attribute name with the `weight` keyword argument. This can be
    used to get the center of mass for instance; assuming the mass of the
    underlying atoms is stored under the "mass" attribute, setting `weight` to
    "mass" will place the bead at the center of mass. By default, `weight` is
    set to `None` and the center of geometry is used.

    The atoms in the underlying graph must have a position. If they do not,
    they are ignored from the average.

    Parameters
    ----------
    molecule: vermouth.molecule.Molecule
        The molecule to update. The attribute `position` of the particles
        is updated on place. The nodes of the molecule must have an attribute
        `graph` that contains the subgraph of the initial molecule.
    ignore_missing_graphs: bool
        If `True`, skip the atoms that do not have a `graph` attribute; else
        fail if not all the atoms in the molecule have a `graph` attribute.
    weight: collections.abc.Hashable
        The name of the attribute used to weight the position of the node. The
        attribute is read from the underlying atoms.
    """
    # Make sure the molecule fullfill the requirements.
    missing = []
    for node in molecule.nodes.values():
        if 'graph' not in node:
            missing.append(node)
        elif weight is not None:
            have_all_weights = all(
                weight in subnode for subnode in node['graph'].nodes.values()
            )
            if not have_all_weights:
                raise KeyError('Not all underlying atoms have an attribute {}.'
                               .format(weight))
    if missing and not ignore_missing_graphs:
        raise ValueError('{} particles are missing the graph attribute'
                         .format(len(missing)))

    for node in molecule.nodes.values():
        if 'graph' in node:
            positions = np.array([
                subnode['position']
                for subnode in node['graph'].nodes().values()
                if subnode.get('position') is not None
            ])
            weights = np.array([
                node.get('mapping_weights', {}).get(subnode_key, 1) * subnode.get(weight, 1)
                for subnode_key, subnode in node['graph'].nodes.items()
                if subnode.get('position') is not None
            ])
            try:
                ndim = positions.shape[1]
            except IndexError:
                ndim = 3
            if abs(sum(weights)) < 1e-7:
                node['position'] = np.array([np.nan]*ndim, dtype=float)
            else:
                node['position'] = np.average(positions, axis=0, weights=weights)

    return molecule


class DoAverageBead(Processor):
    def __init__(self, ignore_missing_graphs=False, weight=None):
        super().__init__()
        self.ignore_missing_graphs = ignore_missing_graphs
        self.weight = weight

    def run_molecule(self, molecule):
        if self.weight is None:
            weight = molecule.force_field.variables.get('center_weight', None)
        elif self.weight is False:
            weight = None
        else:
            weight = self.weight
        return do_average_bead(molecule, self.ignore_missing_graphs, weight=weight)
