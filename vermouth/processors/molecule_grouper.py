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


import collections
import itertools
import networkx as nx
import numpy as np

from scipy.optimize import linprog
from scipy.cluster.vq import kmeans2
from scipy.sparse import csc_matrix, dok_matrix

from .. import KDTree

from ..selectors import is_water, selector_has_position
from .processor import Processor
from ..molecule import Molecule
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


def expand_to_list(num_or_list, length):
    """
    If `num_or_list` is a single item, turns it into a list of length `length`.
    Otherwise, returns `num_or_list`.

    Parameters
    ----------
    num_or_list
    lenght: int

    Returns
    -------
    list
    """
    try:
        len(num_or_list)
    except TypeError:
        num_or_list = [num_or_list] * length
    return num_or_list


def constrained_kmeans(data, num_clusters,
                       clust_sizes=4, tolerances=0, violation_cost=0.5,
                       init_clusters='fixed', precision=1e5, max_iter=100,
                       simplex_method=nx.network_simplex):
    """
    Clusters data in `num_clusters` clusters, where all clusters will contain
    `clust_sizes` \u00b1 `tolerances` items. If this is not possible, an exception
    is raised. If `tolerances` is not 0 clusters can "borrow" or "donate" up to
    `tolerances` items to a virtual data point that is at a distance of
    `violation_cost` units in order to fulfill the constraints.

    Initial cluster placement is controlled by `init_clusters`. If this is
    "fixed" initial clusters will be placed at the origin. If "random", random
    data points will be taken as initial clusters. Otherwise, `init_clusters`
    is taken as initial cluster positions.

    The underlying algorithm (:func:`~networkx.algorithms.flow.network_simplex`)
    requires all arguments to be integers, including the distances. Therefore
    all distances will be multiplied by `precision` before being rounded.

    Loosely based on [1]_.

    Parameters
    ----------
    data: numpy.ndarray
        The data points to cluster.
    num_clusters: int
        The number of clusters to create.
    clust_sizes: int or collections.abc.Sequence[int]
        The required number of items for a cluster, possibly per cluster.
    tolerances: int or collections.abc.Sequence[int]
        The allowed deviation of the number of items for a cluster, possibly per
        cluster.
    violation_cost: float
        The cost (in nm) of having an item too many or few in a cluster.
    init_clusters: numpy.ndarray or str
        The initial cluster positions. If "fixed", all clusters will start at
        the origin. If "random", random data points will be taken as initial
        cluster positions. Otherwise, it should be an array containing the
        initial cluster positions.
    precision: float
        The precision with which to round distances to integers.
    max_iter: int
        The maximum number of iterations.
    simplex_method: collections.abc.Callable
        The method to use to solve the minimum cost flow problem. Should have a
        signature identical to :func:`networkx.algorithms.flow.network_simplex`

    Notes
    -----
    Periodic boundary conditions are not taken into account in distance
    calculations.

    References
    ----------
    .. [1] P. Bradley, K. Bennett, A. Demiriz, Constrained k-means clustering, Microsoft Res. Redmond. 20 (2000) 9.

    Returns
    -------
    float
        The cost, corrected for `precision`.
    numpy.ndarray
        A numpy array of shape (num_clusters, data.shape[-1]) containing the
        cluster positions.
    numpy.ndarray
        A numpy array of shape (data.shape[0], num_clusters) containing 0 and 1,
        with 1 meaning a datapoint contributes to a cluster.
    """
    if data.ndim == 1:
        # np.atleast_2d adds the new dimension in the wrong place.
        data = data[:, np.newaxis]
    clust_sizes = expand_to_list(clust_sizes, num_clusters)
    tolerances = expand_to_list(tolerances, num_clusters)
    if len(clust_sizes) != num_clusters:
        raise IndexError('len(max_clust_sizes) must be num_clusters ({}), but '
                         'is {}'.format(num_clusters, len(clust_sizes)))
    if len(tolerances) != num_clusters:
        raise IndexError('len(tolerances) must be num_clusters ({}), but '
                         'is {}'.format(num_clusters, len(tolerances)))
    if isinstance(init_clusters, str) and init_clusters == 'fixed':
        clusters = np.zeros((num_clusters, data.shape[-1]))
    elif isinstance(init_clusters, str) and init_clusters == 'random':
        rng = np.random.default_rng()
        clusters = rng.choice(data, num_clusters, replace=False, axis=0)
    else:
        clusters = np.broadcast_to(init_clusters, (num_clusters, data.shape[-1]))

    # Time to build the DiGraph on which the minimal flow problem is solved.
    # Note that supply/demand are inverted relative to the paper.
    # The graph basically looks like:
    # Data -> clusters <=> artificial sink
    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(range(len(data) + num_clusters + 1))
    data_nodes = list(range(len(data)))
    clust_nodes = list(range(len(data), len(data) + num_clusters))
    artificial_node = len(data) + num_clusters
    # Demand for artificial node
    rest_cost = len(data) - sum(clust_sizes)
    flow_graph.nodes[artificial_node]['demand'] = rest_cost
    for clust_n_idx in clust_nodes:
        # Edges between clusters and artificial node
        # The paper only has an edge from clusters to artificial node with an
        # infinite capacity and weight 0, since it only deals with a *minimal*
        # number of items per cluster (i.e. you can add more items to any
        # cluster to satisfy the cluster's requirements, and shove any
        # superfluous items on to the artificial node).
        # The capacity means the cluster can donate (or receive) a limited
        # number of supply from the artificial node, and this number exactly is
        # the deviation from the desired number of items per cluster.
        # The weight affects how often/easy it is to deviate from the required
        # number of items per cluster, even when a perfect/equal partitioning is
        # possible.
        flow_graph.add_edge(clust_n_idx, artificial_node,
                            weight=round(precision*violation_cost**2),
                            capacity=tolerances[clust_n_idx-len(data_nodes)])
        flow_graph.add_edge(artificial_node, clust_n_idx,
                            weight=round(precision*violation_cost**2),
                            capacity=tolerances[clust_n_idx-len(data_nodes)])
        # Demand for cluster nodes
        flow_graph.nodes[clust_n_idx]['demand'] = clust_sizes[clust_n_idx - len(data)]
    for data_n_idx in data_nodes:
        # Data supply
        flow_graph.nodes[data_n_idx]['demand'] = -1
    for data_n_idx, clust_n_idx in itertools.product(data_nodes, clust_nodes):
        # Edges between data and clusters
        # Capacity is meaningless, since data nodes have only 1 supply to give,
        # but it seems to help the network simplex a little.
        flow_graph.add_edge(data_n_idx, clust_n_idx, capacity=1)

    # Start clustering iterations.
    iter = 0
    flow_dict = {}
    while iter < max_iter:
        # Step 1: set up edge weights based on clustroid positions
        # TODO: PBC
        dists = np.sum((data[:, np.newaxis] - clusters[np.newaxis, :])**2, axis=-1)
        # When clusters are empty their clustroids become nan, which we can't
        # round to an int.
        dists[np.isnan(dists)] = 10**2
        for (data_idx, clust_idx), dist in np.ndenumerate(dists):
            # Network simplex (and capacity scaling) does not work with float
            # weights according to nx documentation. So round the distances to
            # an appropriate int. Magic.
            flow_graph.edges[data_idx, clust_idx + len(data)]['weight'] = round(precision*dist)

        prev_weights = flow_dict
        cost, flow_dict = simplex_method(flow_graph)

        if prev_weights == flow_dict:
            break
        # Step 2: make memberships and calculate new clustroid positions
        memberships = np.zeros((len(data), num_clusters))
        for data_idx in data_nodes:
            flow = flow_dict[data_idx]
            memberships[data_idx] = [flow[clust_n_idx] for clust_n_idx in clust_nodes]

        # TODO: PBC
        weights = memberships / memberships.sum(axis=0, keepdims=True)
        clusters = weights.T.dot(data)

        iter += 1

    return cost/precision, clusters, memberships, iter


def _assign_members(data, clusters, clust_sizes, tolerances, cutoff=10):
    num_clusters = clusters.shape[0]
    dispersions = data.sparse_distance_matrix(KDTree(clusters), cutoff)
    # Dispersions is: {(at_idx, clust_idx): distance}
    # Let's condense the dispersions to a 1D array. We also want the indices
    # associated with the values. We can't use dispersions.nonzero, since that
    # excludes explicit zeros. And I'd rather not use dispersions.data, since
    # then I have no guarantees over the order of the values.
    disp_idxs = tuple(zip(*dispersions.keys()))
    dispersions = dispersions[disp_idxs].toarray().squeeze()
    disp_idxs = np.array(disp_idxs)
    # We need to make the constraint matrices. Fortunately linprog (or at least
    # the interior point method) can deal with sparse matrices. Saves us a few
    # 10's GB of memory.
    # This one is for the inequality constraints, so that the clusters are
    # between the specified sizes. We need 2*num_clusters, since we need both
    # the upper and lower bound. Indices 0..num_clusters are the upper bounds,
    # indices num_clusters..2*num_clusters lower bounds.
    A_ub = dok_matrix((2*num_clusters, disp_idxs.shape[-1]), dtype=np.int8)
    b_ub = np.empty(2*num_clusters, dtype=np.int8)
    # TODO: Add lower bound so that 90% of all data contributes, and remove the
    #       equality constraint that makes every data point contribute once.
    for jdx in range(num_clusters):
        at_idxs_in_disp = disp_idxs[1] == jdx

        A_ub[jdx, at_idxs_in_disp] = 1
        A_ub[jdx+num_clusters, at_idxs_in_disp] = -1

        b_ub[jdx] = clust_sizes[jdx] + tolerances[jdx]
        num_candidates = np.count_nonzero(at_idxs_in_disp)
        if num_candidates == 0:
            b_ub[jdx + num_clusters] = 0
        elif num_candidates < (clust_sizes[jdx] - tolerances[jdx]):
            # If there are not enough candidate members for a given cluster (due
            # to the distance cutoff in determining the dispersions) the problem
            # can become infeasible. So in those cases, set the lower bound to
            # 1.
            b_ub[jdx + num_clusters] = -1
        else:
            b_ub[jdx+num_clusters] = -(clust_sizes[jdx] - tolerances[jdx])

    # We need to do something similar for all datapoints, to ensure they
    # contribute exactly once.
    A_eq = dok_matrix((data.data.shape[0], disp_idxs.shape[-1]), dtype=np.int8)
    b_eq = np.ones(data.data.shape[0], dtype=np.int8)
    for idx in range(data.data.shape[0]):
        clust_jdxs_in_disp = disp_idxs[0] == idx

        A_eq[idx, clust_jdxs_in_disp] = 1
        if not np.any(clust_jdxs_in_disp):
            # If no candidate clusters the datapoint doesn't have to contribute.
            b_eq[idx] = 0

    # Time so solve the system. Hopefully.
    A_ub = csc_matrix(A_ub)
    A_eq = csc_matrix(A_eq)
    LOGGER.debug('Solving linear problem with {} variables. This may take a '
                 'while...', disp_idxs.shape[1])
    answer = linprog(dispersions, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1),
                     options=dict(sparse=True, tol=0.1))
    members = answer.x
    print(answer)
    # With a bit of luck this is ~0.5, so all memberships are either 0 or 1.
    hardness = np.abs(members - 0.5)
    print(np.min(hardness), np.mean(hardness), np.median(hardness), np.max(hardness))

    memberships = np.zeros((data.data.shape[0], num_clusters))
    print(disp_idxs)
    print(disp_idxs.shape)
    print(dispersions.shape)
    print(members.shape)
    print(data.size)
    print(num_clusters)
    for d_idx in range(disp_idxs.shape[1]):
        idx, jdx = disp_idxs[:, d_idx]
        val = members[d_idx]
        memberships[idx, jdx] = val

    return answer.fun, memberships


def do_cluster(data, num_clusters, clust_sizes=4, tolerances=0,
               init_clusters='random', max_iter=100):
    if data.ndim == 1:
        # np.atleast_2d adds the new dimension in the wrong place.
        data = data[:, np.newaxis]
    clust_sizes = expand_to_list(clust_sizes, num_clusters)
    tolerances = expand_to_list(tolerances, num_clusters)
    if len(clust_sizes) != num_clusters:
        raise IndexError('len(max_clust_sizes) must be num_clusters ({}), but '
                         'is {}'.format(num_clusters, len(clust_sizes)))
    if len(tolerances) != num_clusters:
        raise IndexError('len(tolerances) must be num_clusters ({}), but '
                         'is {}'.format(num_clusters, len(tolerances)))
    if isinstance(init_clusters, str) and init_clusters == 'fixed':
        clusters = np.zeros((num_clusters, data.shape[-1]))
    elif isinstance(init_clusters, str) and init_clusters == 'random':
        rng = np.random.default_rng()
        clusters = rng.choice(data, num_clusters, replace=False, axis=0)
    else:
        clusters = np.broadcast_to(init_clusters, (num_clusters, data.shape[-1]))

    data = KDTree(data)

    dists, members = data.query(clusters, max(s - t for s, t in zip(clust_sizes, tolerances)), eps=0.1)
    clusters = data.data[members].mean(axis=1)
    # dists, members = data.query(clusters, 3, eps=0.1)
    current_iter = 0
    best_fun_val = float('inf')

    while current_iter <= max_iter:
        # We can probably progressively tighten the cutoff?
        funval, memberships = _assign_members(data, clusters, clust_sizes, tolerances, 1)

        if funval >= best_fun_val:
            break
        else:
            best_fun_val = funval

        weights = (0.5 - memberships)**2 / np.sum((0.5 - memberships)**2, axis=0, keepdims=True)
        weights[:, np.all(weights == 0, axis=0)] = np.random.random()
        clusters = weights.T.dot(data.data)
        current_iter += 1
    memberships = memberships.round()
    return best_fun_val, clusters, memberships, current_iter


def group_molecules(system, selector, size_tries=10, clust_sizes=3, **kwargs):
    """
    Clusters molecules in `system` into groups of 4 \u00b1 1. Only molecules
    selected with `selector` will be taken.

    Parameters
    ----------
    system
    selector
    size_tries: int
        The number of clusters to try.
    **kwargs
        Passed on to :func:`constrained_kmeans`.

    Returns
    -------
    None
        `system` is modified in place.
    """
    water_mols = [(mol_idx, mol) for (mol_idx, mol) in enumerate(system.molecules) if selector(mol)]
    if not water_mols:
        return
    mol_idxs, water_mols = zip(*water_mols)
    positions = []
    for mol in water_mols:
        # TODO: Select CoM vs CoG (current) from FF settings/metavars?
        # TODO: What happens if no atom has a position?
        position = np.average([mol.nodes[n_idx]['position']
                               for n_idx in mol
                               if selector_has_position(mol.nodes[n_idx])], axis=0)
        positions.append(position)
    positions = np.array(positions)
    num_clusters = int(np.ceil(len(water_mols)/clust_sizes))
    min_clusters = max(num_clusters - size_tries//2, 0)
    max_clusters = min(num_clusters + size_tries//2, len(positions))
    centroids, labels = kmeans2(positions, k=min_clusters, minit='++')
    members = np.zeros((positions.shape[0], min_clusters))
    for idx in range(min_clusters):
        members[labels==idx, idx] = 1
    weights = members / members.sum(axis=0, keepdims=True)
    init = weights.T.dot(positions)
    # init = kwargs.pop('init_clusters', 'random')
    results = []
    for num_clusters in range(min_clusters, max_clusters+1):
        LOGGER.debug('Trying to cluster {} molecules into {} clusters', len(water_mols), num_clusters)
        # cost, clusters, memberships, niter = constrained_kmeans(
        #     data=positions,
        #     num_clusters=num_clusters,
        #     init_clusters=init,
        #     clust_sizes=clust_sizes,
        #     **kwargs
        # )
        cost, clusters, memberships, niter = do_cluster(
            data=positions,
            num_clusters=num_clusters,
            init_clusters=init,
            clust_sizes=clust_sizes,
            max_iter=3
        )
        init = np.append(clusters, [[0, 0, 0]], axis=0)
        results.append([cost, clusters, memberships, niter])

    cost, clusters, memberships, niter = min(results, key=lambda i: i[0])
    LOGGER.info('Clustered {} water molecules into {} clusters.', len(water_mols), num_clusters)
    counts = np.count_nonzero(memberships, axis=0)
    LOGGER.info('Minimum/maximum number of molecules per cluster: {}/{}', counts.min(), counts.max())
    for mol_idx in sorted(mol_idxs, reverse=True):
        del system.molecules[mol_idx]

    for clust_idx in range(memberships.shape[1]):
        members = memberships[:, clust_idx]
        mols = [mol for (mol, val) in zip(water_mols, members) if val]
        union = Molecule(meta=mols[0].meta, force_field=mols[0].force_field,
                         nrexcl=mols[0].nrexcl)
        previous = []
        for mol_idx, mol in enumerate(mols):
            added = union.merge_molecule(mol)
            for n_idx in added.values():
                union.nodes[n_idx]['mol_idx'] = mol_idx
            previous.append(added.values())
        # Create a "cycle" of the mapped molecules, so the mapping for 3 waters
        # doesn't match 4 waters, etc.
        for idxs, jdxs in zip(previous[:-1], previous[1:]):
            union.add_edges_from(itertools.product(idxs, jdxs))
        union.add_edges_from(itertools.product(previous[0], previous[-1]))
        system.add_molecule(union)


class MoleculeGrouper(Processor):
    def __init__(self, selector=is_water, **cluster_kwargs):
        self.selector = selector
        self.cluster_kwargs = cluster_kwargs

    def run_system(self, system):
        group_molecules(system, self.selector, **self.cluster_kwargs)
