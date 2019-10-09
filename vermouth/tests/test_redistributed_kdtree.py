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
Test the modifications made to the redistributed KDTree.
"""


import hypothesis
from hypothesis import strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from vermouth.redistributed.kdtree import KDTree as redisKDTree

try:
    from scipy.spatial import cKDTree as scipyKDTree
except ImportError:
    # scipy is not available
    HAS_SCIPY = False
else:
    HAS_SCIPY = True


def dict_close(left, right):
    """
    Compare 2 dicts whith floats as values.

    Returns `True` is all the keys are the same and the values are all close
    according to :func:`np.allclose`.

    The order of the keys is not kept.
    """
    if left.keys() != right.keys():
        return False
    keys = sorted(left)
    values_left = np.array([left[key] for key in keys])
    values_right = np.array([right[key] for key in keys])
    return np.allclose(values_left, values_right)


@pytest.mark.skipif(not HAS_SCIPY, reason="Scipy is not available.")
@hypothesis.given(
    coordinates=hnp.arrays(
        dtype=st.sampled_from((np.float32, np.float64)),
        shape=st.tuples(st.integers(1, 10), st.integers(1, 10)),
        elements=st.floats(allow_nan=False, allow_infinity=False, width=32),
    ),
    max_dist=st.floats(0, 5, width=32),
    p=st.integers(1, 5),
)
def test_sparse_distance_matrix(coordinates, max_dist, p):
    """
    Test that the `sparse_distance_matrix` method returns the same thing in the
    redistributed KDTree and on the actual scipy one.
    """
    original_tree = scipyKDTree(coordinates)
    redis_tree = redisKDTree(coordinates)

    original_output = original_tree.sparse_distance_matrix(original_tree, max_dist, p)
    redis_output = redis_tree.sparse_distance_matrix(redis_tree, max_dist, p)

    assert dict_close(redis_output, original_output)
