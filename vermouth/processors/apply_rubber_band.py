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
Provides a processor that adds a rubber band elastic network.
"""
import itertools

import numpy as np
import networkx as nx

from .processor import Processor
from .. import selectors

DEFAULT_BOND_TYPE = 6


def self_distance_matrix(coordinates):
    """
    Compute a distance matrix between points in a selection.

    Notes
    -----
    This function does **not** account for periodic boundary conditions.

    Parameters
    ----------
    coordinates: numpy.ndarray
        Coordinates of the points in the selection. Each row must correspond
        to a point and each column to a dimension.

    Returns
    -------
    numpy.ndarray
    """
    return np.sqrt(
        np.sum(
            (coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2,
            axis=-1)
    )


def compute_decay(distance, shift, rate, power):
    r"""
    Compute the decay function of the force constant as function to the distance.

    The decay function for the force constant is defined as:

    .. math::

        \exp^{-r(d - s)^p}

    where :math:`r` is the decay rate given by the 'rate' argument,
    :math:`p` is the decay power given by 'power', :math:`s` is a shift
    given by 'shift', and :math:`d` is the distance between the two atoms given
    in 'distance'. If the rate or the power are set to 0, then the decay
    function does not modify the force constant.

    The 'distance' argument can be a scalar or a numpy array. If it is an
    array, then the returned value is an array of decay factors with the same
    shape as the input.
    """
    return np.exp(-rate * ((distance - shift) **  power))


def compute_force_constants(distance_matrix, lower_bound, upper_bound,
                            decay_factor, decay_power, base_constant,
                            minimum_force):
    """
    Compute the force constant of an elastic network bond.

    The force constant can be modified with a decay function, and it can be
    bounded with a minimum threshold, or a distance upper and lower bonds.
    """
    constants = compute_decay(distance_matrix, lower_bound, decay_factor, decay_power)
    np.fill_diagonal(constants, 0)
    constants *= base_constant
    constants[constants < minimum_force] = 0
    constants[distance_matrix > upper_bound] = 0
    return constants


def build_connectivity_matrix(graph, separation, selection=None):
    """
    Build a connectivity matrix based on the separation between nodes in a graph.

    The connectivity matrix is a symetric boolean matrix where cells contain
    ``True`` if the corresponding atoms are connected in the graph and
    separated by less or as much nodes as the given 'separation' argument.

    In the following examples, the separation between A and B is 0, 1, and 2.
    respectively:

    ```
    A - B
    A - X - B
    A - X - X - B
    ```

    Note that building the connectivity matrix with a separation of 0 is the
    same as building the adjacency matrix.

    Parameters
    ----------
    graph: networkx.Graph
        The graph/molecule to work on.
    separation: int
        The maximum number of nodes in the shortest path between two nodes of
        interest for these two nodes to be considered connected. Must be >= 0.
    selection: collections.abc.Iterable
        A list of node keys to work on. If this argument is set, then the
        matrix corresponds to the subgraph containing these keys.

    Returns
    -------
    numpy.ndarray
        A boolean matrix.
    """
    if separation < 0:
        raise ValueError('Separation has to be null or positive.')
    if separation == 0:
        # The connectivity matrix with a separation of 1 is the adjacency
        # matrix. Thanksfully, networkx can directly give it to us a a numpy
        # array.
        return nx.to_numpy_matrix(graph, nodelist=selection).astype(bool)
    subgraph = graph.subgraph(selection)
    connectivity = np.zeros((len(subgraph), len(subgraph)), dtype=bool)
    for (idx, key_idx), (jdx, key_jdx) in itertools.combinations(enumerate(subgraph.nodes), 2):
        try:
            shortest_path = len(nx.shortest_path(subgraph, key_idx, key_jdx))
        except nx.NetworkXNoPath:
            # There is no path between key_i and key_j so they are not
            # connected; which is the default.
            pass
        else:
            # The source and the target are counted in the shortest path
            connectivity[idx, jdx] = shortest_path <= separation + 2
            connectivity[jdx, idx] = connectivity[idx, jdx]
    return connectivity


def apply_rubber_band(molecule, selector,
                      lower_bound, upper_bound,
                      decay_factor, decay_power,
                      base_constant, minimum_force,
                      bond_type, res_min_dist=3):
    r"""
    Adds a rubber band elastic network to a molecule.

    The eleastic network is applied as bounds between the atoms selected by the
    function declared with the 'selector' argument. The equilibrium length for
    the bonds is measured from the coordinates in the molecule, the force
    constant is computed from the base force constant and an optional decay
    function.

    The decay function for the force constant is defined as:

    .. math::

        \exp^{-r(d - s)^p}

    where :math:`r` is the decay rate given by the 'decay_factor' argument,
    :math:`p` is the decay power given by 'decay_power', :math:`s` is a shift
    given by 'lower_bound', and :math:`d` is the distance between the two atoms
    in the molecule. If the rate or the power are set to 0, then the decay
    function does not modify the force constant.

    The 'selector' argument takes a callback that accepts a atom dictionary and
    returns ``True`` if the atom match the conditions to be kept.

    Parameters
    ----------
    molecule: Molecule
        The molecule to which apply the elastic network. The molecule is
        modified in-place.
    selector: collections.abc.Callable
        Selection function.
    lower_bound: float
        The minimum length for a bond to be added, expressed in
        nanometers.
    upper_bound: float
        The maximum length for a bond to be added, expressed in
        nanometers.
    decay_factor: float
        Parameter for the decay function.
    decay_power: float
        Parameter for the decay function.
    base_constant: float
        The base force constant for the bonds in :math:`kJ.mol^{-1}.nm^{-2}`.
        If 'decay_factor' or 'decay_power' is set to 0, then it will be the
        used force constant.
    minimum_force: float
        Minimum force constat in :math:`kJ.mol^{-1}.nm^{-2}` under which bonds
        are not kept.
    bond_type: int
        Gromacs bond function type to apply to the elastic network bonds.
    res_min_dist: int
        Minimum separation between two atoms for a bond to be kept.
        Bonds are kept is the separation is greater or equal to the value
        given.
    """
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
    connectivity = build_connectivity_matrix(molecule, res_min_dist - 1,
                                             selection=selection)
    # Set the force constant to 0 for pairs that are connected. `connectivity`
    # is a matrix of booleans that is True when a pair is connected. Because
    # booleans acts as 0 or 1 in operation, we multiply the force constant
    # matrix by the oposite (OR) of the connectivity matrix.
    constants *= ~connectivity
    distance_matrix = distance_matrix.round(5)  # For compatibility with legacy
    for from_idx, to_idx in zip(*np.triu_indices_from(constants)):
        from_key = selection[from_idx]
        to_key = selection[to_idx]
        force_constant = constants[from_idx, to_idx]
        length = distance_matrix[from_idx, to_idx]
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
                 bond_type=None,
                 selector=selectors.select_backbone,
                 bond_type_variable='elastic_network_bond_type'):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.decay_factor = decay_factor
        self.decay_power = decay_power
        self.base_constant = base_constant
        self.minimum_force = minimum_force
        self.bond_type = bond_type
        self.selector = selector
        self.bond_type_variable = bond_type_variable

    def run_molecule(self, molecule):
        # Choose the bond type. From high to low, the priority order is:
        # * what is set as an argument to the processor
        # * what is written in the force field variables
        #   under the key `self.bond_type_variable`
        # * the default value set in DEFAULT_BOND_TYPE
        bond_type = self.bond_type
        if self.bond_type is None:
            bond_type = molecule.force_field.variables.get(self.bond_type_variable,
                                                           DEFAULT_BOND_TYPE)

        apply_rubber_band(molecule, self.selector,
                          lower_bound=self.lower_bound,
                          upper_bound=self.upper_bound,
                          decay_factor=self.decay_factor,
                          decay_power=self.decay_power,
                          base_constant=self.base_constant,
                          minimum_force=self.minimum_force,
                          bond_type=bond_type)
        return molecule
