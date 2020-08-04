# Copyright 2020 University of Groningen
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


from .. import KDTree
from ..selectors import is_water, selector_has_position
from .processor import Processor

import collections
import networkx as nx
import numpy as np
from numbers import Number


def constrained_kmeans(data, num_clusters,
                       max_clust_sizes=4, init_clusters='fixed',
                       precision=1e8):
    """
    Clusters data in num_clusters clusters, where the maximal cluster size is
    constrained by max_clust_sizes. Initial cluster placement is controlled by
    init_clusters. If this is "fixed" initial clusters will be placed at the
    origin. If "random", random data points will be taken as initial clusters.

    Parameters
    ----------
    data: np.ndarray
    num_clusters: int
    max_clust_sizes: number.Number or collections.abc.Sequence[number.Number]
    init_clusters: np.ndarray or str
        If "fixed", ... if "random", ...

    Returns
    -------
    ...
    """
    try:
        len(max_clust_sizes)
    except TypeError:
        # max_clust_sizes is number-like
        max_clust_sizes = [max_clust_sizes] * num_clusters
    if len(max_clust_sizes) != num_clusters:
        raise IndexError('len(max_clust_sizes) must be num_clusters ({}), but '
                         'is {}'.format(num_clusters, len(max_clust_sizes)))
    if init_clusters == 'fixed':
        clusters = np.zeros((num_clusters, data.shape[-1]))
    elif init_clusters == 'random':
        rng = np.random.default_rng()
        clusters = rng.choice(data, num_clusters, replace=False, axis=0)
    else:
        clusters = init_clusters

    # Time to build the DiGraph on which the minimal flow problem is solved.
    # Note that supply/demand are inverted relative to the paper.
    # The graph basically looks like:
    # Data -> clusters -> artificial sink
    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(range(len(data) + num_clusters + 1))
    data_nodes = list(range(len(data)))
    clust_nodes = list(range(len(data), len(data) + num_clusters))
    artificial_node = len(data) + num_clusters
    # Demand for artificial node
    flow_graph.nodes[artificial_node]['demand'] = len(data) - sum(max_clust_sizes)
    for clust_n_idx in clust_nodes:
        # Edges between clusters and artificial node
        flow_graph.add_edge(clust_n_idx, artificial_node, weight=0)
        # Demand for cluster nodes
        flow_graph.nodes[clust_n_idx]['demand'] = max_clust_sizes[clust_n_idx - len(data)]
        for data_n_idx in data_nodes:
            # Edges between data and clusters
            flow_graph.add_edge(data_n_idx, clust_n_idx)
            # Data supply
            flow_graph.nodes[data_n_idx]['demand'] = -1

    # Start clustering iterations.
    iter = 0
    max_iter = 100
    prev_weights = None
    flow_dict = -1
    while iter < max_iter:
        # Step 1: set up edge weights based on clustroid positions
        # TODO: PBC
        dists = np.sum((data[:, np.newaxis] - clusters[np.newaxis, :])**2, axis=-1)
        for (data_idx, clust_idx), dist in np.ndenumerate(dists):
            # Capacity scaling does not work with float weights according to nx
            # documentation. Also makes network simplex work. Magic.
            flow_graph.add_edge(data_idx, clust_idx + len(data), weight=round(precision*dist))

        prev_weights = flow_dict
        _, flow_dict = nx.network_simplex(flow_graph)

        if prev_weights == flow_dict:
            break

        # Step 2: calculate new clustroid positions
        memberships = np.zeros((len(data), num_clusters))
        for data_idx, flow in flow_dict.items():
            if data_idx >= len(data):  # cluster and artificial nodes
                continue
            memberships[data_idx] += [flow[clust_n_idx] for clust_n_idx in clust_nodes]
        # TODO: PBC
        clusters = memberships.T.dot(data)

        iter += 1
    print(iter)
    print(clusters)
    print(memberships)
    return memberships, clusters


def group_molecules(system, selector):
    water_mols = [mol for mol in system.molecules if selector(mol)]
    positions = []
    for mol in water_mols:
        # TODO: Select CoM vs CoG (current) from FF settings/metavars
        position = np.average([mol.nodes[n_idx]['position']
                               for n_idx in mol
                               if selector_has_position(mol.nodes[n_idx])])
        positions.append(position)
    memberships = constrained_kmeans(positions, ...)


class MoleculeGrouper(Processor):
    def __init__(self):
        self.selector = is_water

    def run_system(self, system):
        group_molecules(system, self.selector)