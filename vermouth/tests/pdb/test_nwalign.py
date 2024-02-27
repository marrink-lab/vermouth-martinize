# -*- coding: utf-8 -*-
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
Unittests for the PDB reader.
"""

import numpy as np
import pytest
from vermouth.pdb.nwalign import *


def test_OLA():
    assert OLA_codes(['ALA', 'GLY', 'PHE']) == 'AGF'

@pytest.mark.parametrize('alignment_indicator, output', (
        (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), None),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), None),
        (np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]), ([0], [3])),
        (np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]), ([3], [9])),
        (np.array([1, 1, 0, 0, 1, 1, 0, 0, 1]), ([2, 6], [4, 8])),
        (np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]), ([0, 6], [3, 9])),
))
def test_res_matching(alignment_indicator, output):
    test_output = res_matching(alignment_indicator)
    if test_output is not None:
        assert test_output == output
    else:
        assert True