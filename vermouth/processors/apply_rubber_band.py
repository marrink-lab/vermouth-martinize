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
import networkx as nx
from .processor import Processor
from .. import selectors


def self_distance_matrix(coordinates):
    return np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(axis=-1))


def compute_decay(distance_matrix, shift, rate, power):
    return np.exp(-rate * ((distance_matrix - shift) **  power))


def compute_force_constants(distance_matrix, lower_bound, upper_bound,
                            decay_factor, decay_power, base_constant,
                            minimum_force):
    constants = compute_decay(distance_matrix, lower_bound, decay_factor, decay_power)
    np.fill_diagonal(constants, 0)
    constants *= base_constant
    constants[constants < minimum_force] = 0
    constants[distance_matrix > upper_bound] = 0
    return constants


def apply_rubber_band(molecule, selector,
                      lower_bound, upper_bound,
                      decay_factor, decay_power,
                      base_constant, minimum_force,
                      bond_type):
    selection = []
    coordinates = []
    missing = []
    for node_key, attributes in molecule.nodes.items():
        if selector(attributes):
            selection.append(node_key)
            coordinates.append(attributes.get('position'))
            if coordinates[-1] is None:
                missing.append(node_key)
    if missing:
        raise ValueError('All atoms from the selection must have coordinates. '
                         'The following atoms do not have some: {}.'
                         .format(' '.join(missing)))
    coordinates = np.stack(coordinates)
    distance_matrix = self_distance_matrix(coordinates)
    constants = compute_force_constants(distance_matrix, lower_bound,
                                        upper_bound, decay_factor, decay_power,
                                        base_constant, minimum_force)
    connectivity = nx.to_numpy_matrix(molecule, nodelist=selection).astype(bool)
    constants *= ~connectivity
    for from_idx, to_idx in zip(*np.triu_indices_from(constants)):
        from_key = selection[from_idx]
        to_key = selection[to_idx]
        force_constant = constants[from_idx, to_idx]
        length = round(distance_matrix[from_idx, to_idx], 5)
        if force_constant > minimum_force:
            molecule.add_interaction(
                type_='bonds',
                atoms=(from_key, to_key),
                parameters=[bond_type, length, force_constant],
                meta={'group': 'Rubber band'},
            )


class ApplyRubberBand(Processor):
    def __init__(self, lower_bound, upper_bound, decay_factor, decay_power,
                 base_constant, minimum_force,
                 bond_type=6,
                 selector=selectors.select_backbone):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.decay_factor = decay_factor
        self.decay_power = decay_power
        self.base_constant = base_constant
        self.minimum_force = minimum_force
        self.bond_type = bond_type
        self.selector = selector

    def run_molecule(self, molecule):
        apply_rubber_band(molecule, self.selector,
                          lower_bound=self.lower_bound,
                          upper_bound=self.upper_bound,
                          decay_factor=self.decay_factor,
                          decay_power=self.decay_power,
                          base_constant=self.base_constant,
                          minimum_force=self.minimum_force,
                          bond_type=self.bond_type)
        return molecule
