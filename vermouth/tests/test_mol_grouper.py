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

from hypothesis import given, settings, note
from hypothesis import strategies as st
import hypothesis.extra.numpy as npst

from functools import partial
import numpy as np
from vermouth.processors.molecule_grouper import constrained_kmeans, expand_to_list


def _is_valid_clust_tol(clust_spec, npoints, n_clusters):
    clust_size, tolerance = clust_spec
    clust_size = expand_to_list(clust_size, n_clusters)
    tolerance = expand_to_list(tolerance, n_clusters)

    low = 0
    high = 0
    for size, tol in zip(clust_size, tolerance):
        low += max(size - tol, 0)
        high += size + tol

    return low <= npoints <= high


def finite_real_arrays(shape):
    return npst.arrays(dtype=float,
                       elements=st.floats(min_value=-1e8, max_value=1e8,
                                          allow_nan=False, allow_infinity=False),
                       shape=shape)

@settings(deadline=None)
@given(st.data())
def test_constrained_kmeans(data):

    dims = st.integers(min_value=1, max_value=3)
    num_points = st.integers(min_value=1, max_value=50)
    point_shape = data.draw(st.tuples(num_points, dims), label='point_shape')
    points = finite_real_arrays(point_shape)
    n_points = point_shape[0]
    n_clusters = data.draw(st.integers(min_value=1, max_value=n_points), label='n_clusters')

    clust_sizes = st.one_of(st.integers(min_value=0, max_value=n_points),
                            st.lists(st.integers(min_value=0, max_value=n_points),
                                     min_size=n_clusters, max_size=n_clusters))
    tolerances = st.one_of(st.integers(min_value=0),
                           st.lists(st.integers(min_value=0),
                                    min_size=n_clusters, max_size=n_clusters))
    clust_specs = st.tuples(clust_sizes, tolerances).filter(
        partial(_is_valid_clust_tol, npoints=n_points, n_clusters=n_clusters)
    )

    inits = st.one_of(
            st.just('random'),
            st.just('fixed'),
            finite_real_arrays((n_clusters, point_shape[1]))
        )

    point, clust_spec, init = data.draw(st.tuples(points, clust_specs, inits))
    clust_size, tolerance = clust_spec

    cost, clusters, memberships, iter = constrained_kmeans(point, n_clusters,
                                                           clust_sizes=clust_size,
                                                           tolerances=tolerance,
                                                           init_clusters=init)
    assert memberships.sum() == n_points  # Every point gets assigned
    assert np.all(memberships.sum(axis=1) == 1)  # Every point gets assigned once
    assert clusters.shape == (n_clusters, point_shape[1])
    assert np.allclose(memberships.T.dot(point)/memberships.sum(axis=0, keepdims=True).T, clusters, equal_nan=True)
    assert np.all(memberships.sum(axis=0) <= np.array(clust_size) + tolerance)
    assert np.all(memberships.sum(axis=0) >= np.array(clust_size) - tolerance)
