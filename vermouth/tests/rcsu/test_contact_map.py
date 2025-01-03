# -*- coding: utf-8 -*-
# Copyright 2025 University of Groningen
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
Unit tests for the Go contact map generator.
"""
import pytest
import numpy as np
from vermouth.rcsu import contact_map
from collections import defaultdict
from vermouth.graph_utils import make_residue_graph
from vermouth.rcsu.contact_map import GenerateContactMap
import vermouth
from vermouth.tests.helper_functions import test_molecule, equal_graphs

@pytest.mark.parametrize('resname, atomname, expected',
                         (
                                 ('TRP', 'N', 1.64),
                                 ('TRP', 'NON', 0.00),
                                 ('NON', 'N', 0.00),
                                 ('NON', 'NON', 0.00)
                         ))
def test_get_vdw_radius(resname, atomname, expected):
    result = contact_map._get_vdw_radius(resname, atomname)
    assert result == expected


@pytest.mark.parametrize('resname, atomname, expected',
                         (
                                 ('TRP', 'N', 3),
                                 ('TRP', 'NON', 0),
                                 ('NON', 'N', 0),
                                 ('NON', 'NON', 0)
                         ))
def test_get_atype(resname, atomname, expected):
    result = contact_map._get_atype(resname, atomname)
    assert result == expected

def test_surface_generation():

    position = np.array([1, 1, 1])

    surface = contact_map._make_surface(position, 13, 21, 1)

    assert len(surface) == 21

    first_point = [1.        , 1.        , 2.        ]
    last_point = [1.42591771, 1.        , 0.0952381 ]

    test_points = [first_point, last_point]

    for i, j in enumerate([surface[0], surface[-1]]):
        assert j == pytest.approx(test_points[i])


@pytest.mark.parametrize('norm, expected', (
     (False,
       np.array([[1, 0, 0, 0, 0],
                [0, 8, 0, 0, 0],
                [0, 0, 27, 0, 0],
                [0, 0, 0, 36, 0],
                [0, 0, 0, 0, 80]]
                )
      ),
     (True,
      np.array([[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]]
               )
      )
    ))
def test_atom2res(norm, expected):

    arrin = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
                      ])

    nres = 5
    res_map = contact_map.make_atom_map([0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    res_array = contact_map.atom2res(arrin, nres, res_map, norm)

    assert np.allclose(res_array, expected)

def test_make_atom_map():

    input = [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    result = contact_map.make_atom_map(input)

    expected = defaultdict(list, {0: np.array([0]),
                                  1: np.array([1, 2]),
                                  2: np.array([3, 4, 5]),
                                  3: np.array([6, 7, 8]),
                                  4: np.array([9, 10, 11, 12])})

    for key, value in result.items():
        assert key in expected.keys()


def test_contact_info(test_molecule):
    result = contact_map._contact_info(test_molecule)

    vdw_list, atypes, coords, res_serial, resids, chains, resnames, res_idx, ca_pos, nresidues, G = result

    assert nresidues == 4

    assert np.allclose(vdw_list, [0]*len(test_molecule))
    assert np.allclose(atypes, [0]*len(test_molecule))

    assert np.allclose(coords, np.stack([test_molecule.nodes[node]['position']*10 for node in sorted(test_molecule.nodes)]))

    assert np.allclose(res_serial, np.array([0, 0, 0, 1, 1, 2, 3, 3, 3]))
    assert np.allclose(resids, np.array([1, 2, 3, 4]))
    assert np.allclose(res_idx, np.array([0, 1, 2, 3]))

    assert len(ca_pos) == 0

    assert set(chains) & set(['A']*nresidues)
    assert set(resnames) & set(['res0', 'res1', 'res2', 'res3'])

    assert equal_graphs(make_residue_graph(test_molecule), G)

