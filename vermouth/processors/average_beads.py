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

import numpy as np
from .processor import Processor


def do_average_bead(molecule, ignore_missing_graphs=False):
    """
    Set the position of the particles to the mean of the underlying atoms.

    Parameters
    ----------
    molecule: vermouth.Molecule
        The molecule to update. The attribute :attr:`position` of the particles
        is updated on place. The nodes of the molecule must have an attribute
        :attr:`graph` that contains the subgraph of the initial molecule.
    """
    # Make sure the molecule fullfill the requirements.
    missing = []
    for node in molecule.nodes.values():
        if 'graph' not in node:
            missing.append(node)
    if missing and not ignore_missing_graphs:
        raise ValueError('{} particles are missing the graph attribute'
                         .format(len(missing)))

    for node in molecule.nodes.values():
        if 'graph' in node:
            positions = np.stack([
                subnode['position']
                for subnode in node['graph'].nodes().values()
                if 'position' in subnode
            ])
            node['position'] = positions.mean(axis=0)

    return molecule


class DoAverageBead(Processor):
    def __init__(self, ignore_missing_graphs=False):
        super().__init__()
        self.ignore_missing_graphs = ignore_missing_graphs

    def run_molecule(self, molecule):
        return do_average_bead(molecule, self.ignore_missing_graphs)
