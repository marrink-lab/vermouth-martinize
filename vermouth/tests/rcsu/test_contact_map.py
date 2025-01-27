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
from scipy.spatial import cKDTree as KDTree
from vermouth.rcsu.contact_map import GenerateContactMap
from vermouth.tests.helper_functions import test_molecule, equal_graphs, create_sys_all_attrs
from vermouth.tests.datafiles import TEST_MOLECULE_CONTACT_MAP
from vermouth.file_writer import DeferredFileWriter

@pytest.mark.parametrize('resname, atomname, expected',
                         (
                                 ('TRP', 'N', 1.64),
                                 ('TRP', 'NON', 0.00),
                                 ('NON', 'N', 0.00),
                                 ('NON', 'NON', 0.00)
                         ))
def test_get_vdw_radius(resname, atomname, expected):
    # test that we get the correct vdw radii and errors in resname/atomname are handled correctly
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
    # test that a surface is generated with the correct points

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
    # test that atomic resolution arrays get mapped correctly to their residues and are normalised if required

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
    # test that the atom_map defaultdict is generated correctly

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
    # test we get the expected input data from a molecule

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


def test_calculate_overlap(test_molecule):
    # test the overlap is calculated correctly

    result = contact_map._contact_info(test_molecule)
    points = result[2]
    tree = KDTree(points)
    vdw_list = [7] * len(points)
    natoms = len(points)
    vdw_max = 20
    alpha = 1

    overlaps = contact_map._calculate_overlap(tree, vdw_list, natoms, vdw_max, alpha)

    expected = np.array([[0., 1., 1., 1., 1., 1., 0., 0., 0.],
                         [1., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [1., 1., 0., 1., 1., 0., 0., 0., 0.],
                         [1., 1., 1., 0., 1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0., 1., 1., 1., 1.],
                         [1., 1., 0., 1., 1., 0., 1., 1., 1.],
                         [0., 0., 0., 1., 1., 1., 0., 1., 1.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 1.],
                         [0., 0., 0., 0., 1., 1., 1., 1., 0.]])

    assert np.allclose(overlaps, expected)


def test_calculate_csu(test_molecule):
    # test that the csu contacts are found correctly

    result = contact_map._contact_info(test_molecule)
    points = result[2]

    vdw_list = [7] * len(points)
    fiba, fibb = 13, 21
    natoms = len(points)
    tree = KDTree(points)
    vdw_max = 20
    water_radius = 1

    csu_contacts = contact_map._calculate_csu(points,
                                              vdw_list,
                                              fiba,
                                              fibb,
                                              natoms,
                                              tree,
                                              vdw_max,
                                              water_radius)

    # these are the values that sphere points might have contacts with for each point above.
    expected = np.array([[-1,  1,  3, -1,  1, -1,  1,  3, -1,  1,  3,  1,  3, -1,  1,  3, -1,  1, -1,  1,  3],
                         [-1,  2,  0, -1,  2,  0,  2,  4,  0,  2,  0,  2,  4,  0,  2,  0, -1,  2,  0,  2,  4],
                         [-1, -1,  1, -1, -1,  1, -1,  4,  1, -1,  1, -1,  4,  1, -1,  1, -1, -1,  1, -1, -1],
                         [-1,  4,  5,  0,  4, -1,  0,  5,  0,  4,  5,  0,  5,  0,  4,  5, 0,  4, -1,  0,  5],
                         [-1, -1,  3,  1,  7,  3,  1,  5,  1,  8,  3,  1,  5,  1,  2,  3, 1,  8,  3,  1, -1],
                         [-1, -1,  6,  3,  6, -1,  3,  6,  3,  6,  6,  3,  6,  3,  4,  6, 3,  6, -1,  3,  6],
                         [-1,  7, -1,  5,  7, -1,  5, -1,  5,  7, -1,  5, -1,  5,  7, -1, 5,  7, -1,  5, -1],
                         [-1,  8,  6,  4,  8,  6,  8, -1,  6,  8,  6,  8, -1,  6,  8,  6, 5,  8,  6,  8, -1],
                         [-1, -1,  7, -1, -1,  7, -1, -1,  7, -1,  7, -1, -1,  7, -1,  7, 4, -1,  7, -1, -1]])

    assert np.allclose(csu_contacts, expected)

def test_contact_types(test_molecule):

    result = contact_map._contact_info(test_molecule)
    points = result[2]

    vdw_list = [7] * len(points)
    fiba, fibb = 13, 21
    natoms = len(points)
    tree = KDTree(points)
    vdw_max = 20
    water_radius = 1

    hits = contact_map._calculate_csu(points,
                                      vdw_list,
                                      fiba,
                                      fibb,
                                      natoms,
                                      tree,
                                      vdw_max,
                                      water_radius)


    natoms = len(hits)
    # this is slightly dodgy, but we get away with it because it's < 10. gives some variety at least
    atypes = np.arange(natoms)

    expected = [np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 8., 0., 3., 0., 0., 0., 0.],
                          [0., 7., 0., 0., 2., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 5., 6., 0., 0., 0.],
                          [0., 7., 1., 5., 0., 2., 0., 1., 2.],
                          [0., 0., 0., 7., 1., 0., 9., 0., 0.],
                          [0., 0., 0., 0., 0., 7., 0., 5., 0.],
                          [0., 0., 0., 0., 1., 1., 7., 0., 8.],
                          [0., 0., 0., 0., 1., 0., 0., 7., 0.]]),
                np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 8., 0., 0., 0., 0., 0., 0.],
                          [0., 7., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 2., 0., 0., 0.],
                          [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
                np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 3., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 2., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 5., 6., 0., 0., 0.],
                          [0., 7., 1., 5., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 7., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0.]])]

    contact_arrays = contact_map._contact_types(hits, natoms, atypes)

    for i, j in enumerate(contact_arrays):
        assert np.allclose(j, expected[i])


def test_calculate_contacts(test_molecule):

    result = contact_map._contact_info(test_molecule)
    points = result[2]
    res_serial = result[3]
    nresidues = result[9]

    vdw_list = [7] * len(points)
    atypes = np.arange(len(points))

    expected = [np.array([[0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.]]),
                np.array([[449.,  66.,   0.,   0.],
                          [227., 287., 146.,   0.],
                          [  0., 227.,   0., 226.],
                          [  0.,   0., 227., 817.]]),
                np.array([[449.,   0.,   0.,   0.],
                          [  0.,   0.,   0.,   0.],
                          [  0.,   0.,   0.,   0.],
                          [  0.,   0.,   0.,   0.]]),
                np.array([[  0.,  66.,   0.,   0.],
                          [227., 287., 146.,   0.],
                          [  0., 227.,   0.,   0.],
                          [  0.,   0.,   0.,   0.]])]


    contact_arrays = contact_map._calculate_contacts(vdw_list, atypes, points, res_serial, nresidues)

    for i, j in enumerate(contact_arrays):
        assert np.allclose(j, expected[i])

def test_get_contacts(test_molecule):

    result = contact_map._contact_info(test_molecule)
    points = result[2]
    res_serial = result[3]
    res_idx = result[7]
    nresidues = result[9]
    molecule_graph = result[10]

    vdw_list = [7] * len(points)
    atypes = np.arange(len(points))

    overlaps, contacts, stabilisers, destabilisers = contact_map._calculate_contacts(vdw_list, atypes, points,
                                                                                     res_serial, nresidues)

    # add something interesting to overlaps otherwise we get nothing
    overlaps = np.array([[0., 1., 0., 0.],
                         [0., 1., 1., 0.],
                         [0., 1., 1., 0.],
                         [0., 0., 1., 0.]])

    contacts_list, all_contacts = contact_map._get_contacts(nresidues, overlaps, contacts, stabilisers, destabilisers,
                                                            res_idx, molecule_graph)

    expected_all_contacts = [[1, 2, 0, 1, 1.0, 66.0, 0.0, False],
                             [2, 1, 1, 0, 0.0, 227.0, 0.0, False],
                             [2, 3, 1, 2, 1.0, 146.0, 0.0, False],
                             [3, 2, 2, 1, 1.0, 227.0, 0.0, False],
                             [3, 4, 2, 3, 0.0, 226.0, 0.0, False],
                             [4, 3, 3, 2, 1.0, 227.0, 0.0, False]]

    expected_contacts_list = [[1, "A", 2, "A"],
                              [2, "A", 3, "A"],
                              [3, "A", 2, "A"],
                              [4, "A", 3, "A"]]
    for i, j in zip([contacts_list, all_contacts], [expected_contacts_list, expected_all_contacts]):
        for k, l in zip(i, j):
            assert list(k) == list(l)

def test_write_contacts(test_molecule, tmp_path):

    result = contact_map._contact_info(test_molecule)
    points = result[2]
    res_serial = result[3]
    res_idx = result[7]
    nresidues = result[9]
    molecule_graph = result[10]
    vdw_list = [7] * len(points)
    atypes = np.arange(len(points))

    # make some fake ca positions from the COGs of the residues
    ca_pos = []
    for residue in molecule_graph.nodes:
        subgraph = molecule_graph.nodes[residue]['graph']
        pos = []
        for atom in sorted(subgraph.nodes):
            pos.append(subgraph.nodes[atom]['position']*10)
        ca_pos.append(np.mean(np.stack(pos), axis=0))

    overlaps, contacts, stabilisers, destabilisers = contact_map._calculate_contacts(vdw_list, atypes, points,
                                                                                     res_serial, nresidues)

    _, all_contacts = contact_map._get_contacts(nresidues, overlaps, contacts, stabilisers, destabilisers,
                                                res_idx, molecule_graph)

    with open(TEST_MOLECULE_CONTACT_MAP) as expectedfile:
        expected_lines = expectedfile.readlines()

    outpath = tmp_path / 'contacts.out'

    contact_map._write_contacts(outpath,
                                all_contacts,
                                ca_pos,
                                molecule_graph)
    DeferredFileWriter().write()

    with open(str(outpath)) as infile:
        written_lines = infile.readlines()

    # skip the first line here because it's the vermouth version
    for line, expected_line in zip(written_lines[1:], expected_lines[1:]):
        assert line == expected_line


@pytest.mark.parametrize('write_out',
                         (False, True)
                         )
def test_do_contacts(test_molecule, tmp_path, write_out):

    if write_out:
        outpath = str(tmp_path / 'contacts.out')
    else:
        outpath = False

    contacts = contact_map.do_contacts(test_molecule, outpath)

    # because of the vdw radii we actually expect no contacts
    assert len(contacts) == len([]) == 0

    if write_out:
        DeferredFileWriter().write()

        with open(TEST_MOLECULE_CONTACT_MAP) as expectedfile:
            expected_lines = expectedfile.readlines()

        with open(str(outpath)) as infile:
            written_lines = infile.readlines()

        # skip the first line here because it's the vermouth version
        # this will actually just check that the header is written because no contacts are found
        for line, expected_line in zip(written_lines[1:], expected_lines[1:]):
            assert line == expected_line

def test_processor(test_molecule):
    # these don't actually matter because the processor only need coordinates
    atypes = {0: "P1", 1: "SN4a", 2: "SN4a",
              3: "SP1", 4: "C1",
              5: "TP1",
              6: "P1", 7: "SN3a", 8: "SP4"}
    secstruc = {1: "H", 2: "H", 3: "H", 4: "H"}

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"atype": atypes})
    assert len(system.go_params["go_map"]) == 0

    processor = GenerateContactMap(write_file=False)
    processor.run_system(system)

    # really just matters that after the processor has been run a list has been added to the dictionary
    assert len(system.go_params["go_map"]) == 1

