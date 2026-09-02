# Copyright 2022 University of Groningen
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
Test for the tune idp bonds processor.
"""
import logging

import pytest
from vermouth.dssp import dssp
from vermouth.forcefield import ForceField
from vermouth.molecule import Molecule
from vermouth.processors.annotate_idrs import AnnotateIDRs, parse_residues
from vermouth.system import System
from vermouth.tests.datafiles import FF_UNIVERSAL_TEST
from vermouth.tests.helper_functions import create_sys_all_attrs, test_molecule

@pytest.mark.parametrize('idr_regions, expected', [
    (
            ["1:2"],
            [True, True, True, True, True, False, False, False, False]
    ),
    (
            ["0:1", "4:5"],
            [True, True, True, False, False, False, True, True, True,]
    ),
    (
    [],
    [False, False, False, False, False, False, False, False, False]
    )
])
def test_make_disorder_string(test_molecule,
                              idr_regions,
                              expected):
    atypes = {0: "P1", 1: "SN4a", 2: "SN4a",
              3: "SP1", 4: "C1",
              5: "TP1",
              6: "P1", 7: "SN3a", 8: "SP4"}
    # the molecule resnames
    resnames = {0: "A", 1: "A", 2: "A",
                3: "B", 4: "B",
                5: "C",
                6: "D", 7: "D", 8: "D"}
    secstruc = {1: "H", 2: "H", 3: "H", 4: "H"}

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes})

    AnnotateIDRs(id_regions=idr_regions).run_system(system)
    result = []
    for key, node in system.molecules[0].nodes.items():
        if system.molecules[0].nodes[key].get("cgidr"):
            result.append(True)
        else:
            result.append(False)
    print(result)
    print(expected)
    assert result == expected

@pytest.mark.parametrize('idr_regions, secstruc, write_sec, expected',(
        (["1:4"],
         {1: "H", 2: "H", 3: "H", 4: "H"},
         True,
         [{0: "C", 1: "C", 2: "C",
          3: "C", 4: "C",
          5: "C",
          6: "C", 7: "C", 8: "C"}, True]),
        (["1:2"],
         {1: "H", 2: "H", 3: "H", 4: "H"},
         True,
         [{0: "C", 1: "C", 2: "C",
          3: "C", 4: "C",
          5: "H",
          6: "H", 7: "H", 8: "H"}, True]),
        (["1:2"],
         {1: "H", 2: "H", 3: "H", 4: "H"},
         False,
         [{0: None, 1: None, 2: None,
          3: None, 4: None,
          5: None,
          6: None, 7: None, 8: None}, False]),
        (["1:2"],
         {1: "C", 2: "C", 3: "C", 4: "C"},
         True,
         [{0: "C", 1: "C", 2: "C",
          3: "C", 4: "C",
          5: "C",
          6: "C", 7: "C", 8: "C"}, False]),

))
def test_ss_reassign(test_molecule, idr_regions, secstruc, write_sec, expected):
    resnames = {0: "A", 1: "A", 2: "A",
                3: "B", 4: "B",
                5: "C",
                6: "D", 7: "D", 8: "D"}
    atypes = {0: "P1", 1: "SN4a", 2: "SN4a",
              3: "SP1", 4: "C1",
              5: "TP1",
              6: "P1", 7: "SN3a", 8: "SP4"}

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes},
                                  write_secstruct=write_sec)

    AnnotateIDRs(id_regions=idr_regions).run_system(system)

    for key, node in system.molecules[0].nodes.items():
        assert system.molecules[0].nodes[key].get("cgsecstruct", None) == expected[0][key]

    assert system.molecules[0].meta.get("modified_cgsecstruct", False) == expected[1]

@pytest.mark.parametrize('modify, expected',
                         ((True, True),
                         (False, False)
))
def test_gmx_system_header_supplementary(test_molecule, modify, expected):

    atypes = {0: "P1", 1: "SN4a", 2: "SN4a",
              3: "SP1", 4: "C1",
              5: "TP1",
              6: "P1", 7: "SN3a", 8: "SP4"}
    resnames = {0: "ALA", 1: "ALA", 2: "ALA",
                3: "GLY", 4: "GLY",
                5: "MET",
                6: "ARG", 7: "ARG", 8: "ARG"}
    secstruc ={1: "H", 2: "H", 3: "H", 4: "H"}

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes})
    if modify:
        AnnotateIDRs(id_regions=["1:2"]).run_system(system)

    dssp.AnnotateResidues(attribute="aasecstruct",
                          sequence="HHHH").run_system(system)
    dssp.AnnotateMartiniSecondaryStructures().run_system(system)

    assert expected == any(["IDR" in i for i in system.meta.get('header', [''])])

@pytest.mark.parametrize('resspec, expected',
                         ((['A-10:20'],
                           [{'chain': 'A', 'resids': [(10, 20)]}]),
                          (['10:20'],
                           [{'chain': None, 'resids': [(10, 20)]}]),
                         (['10:20', 'A-50:65'],
                          [{'chain': None, 'resids': [(10, 20)]}, {'chain': 'A', 'resids': [(50, 65)]}])

                          ))
def test_parse_disorder_resspec(resspec, expected):
    parsed = []
    for spec in resspec:
        parsed.append(parse_residues(spec))
    assert len(parsed) == len(expected)

    for i,j in zip(parsed, expected):
        for key in i.keys():
            assert i[key] == j[key]

def _make_idr_system():
    """
    Build a minimal system with resids 1-4 in chain A and resids 1-2 in chain B.
    """
    system = System(force_field=ForceField(FF_UNIVERSAL_TEST))
    mol = Molecule(force_field=ForceField(FF_UNIVERSAL_TEST))
    nodes = [
        {'chain': 'A', 'resid': 1, 'stash': {'resid': 1}},
        {'chain': 'A', 'resid': 2, 'stash': {'resid': 2}},
        {'chain': 'A', 'resid': 3, 'stash': {'resid': 3}},
        {'chain': 'A', 'resid': 4, 'stash': {'resid': 4}},
        {'chain': 'B', 'resid': 1, 'stash': {'resid': 1}},
        {'chain': 'B', 'resid': 2, 'stash': {'resid': 2}},
    ]
    mol.add_nodes_from(enumerate(nodes))
    system.add_molecule(mol)
    return system


@pytest.mark.parametrize('id_regions, expected_cgidr', [
    # All residues requested are present in the structure. A region without
    # a chain specifier annotates all chains.
    (["1:4"], {('A', 1): True, ('A', 2): True, ('A', 3): True, ('A', 4): True,
               ('B', 1): True, ('B', 2): True}),
    # Only a subset of the requested residues is present.
    (["3:9"], {('A', 1): False, ('A', 2): False, ('A', 3): True, ('A', 4): True,
               ('B', 1): False, ('B', 2): False}),
    # None of the requested residues are present.
    (["9:11"], {('A', 1): False, ('A', 2): False, ('A', 3): False, ('A', 4): False,
                ('B', 1): False, ('B', 2): False}),
    # Multiple regions, one fully present and one fully absent.
    (["1:2", "9:11"], {('A', 1): True, ('A', 2): True, ('A', 3): False, ('A', 4): False,
                       ('B', 1): True, ('B', 2): True}),
    # Chain-specific region: only chain A is annotated.
    (["A-1:4"], {('A', 1): True, ('A', 2): True, ('A', 3): True, ('A', 4): True,
                 ('B', 1): False, ('B', 2): False}),
    # Chain-specific region: only chain B is annotated.
    (["B-1:2"], {('A', 1): False, ('A', 2): False, ('A', 3): False, ('A', 4): False,
                 ('B', 1): True, ('B', 2): True}),
])
def test_annotate_idr_regions(id_regions, expected_cgidr):
    """
    Tests that the cgidr annotation is set on the residues that are
    present in the structure and requested via -id-regions.
    """
    system = _make_idr_system()

    AnnotateIDRs(id_regions=id_regions).run_system(system)

    for node, (chain, resid) in enumerate(
            [('A', 1), ('A', 2), ('A', 3), ('A', 4), ('B', 1), ('B', 2)]):
        assert system.molecules[0].nodes[node].get('cgidr') == expected_cgidr[(chain, resid)]


@pytest.mark.parametrize('id_regions, expected', [
    # All residues requested are present in the structure.
    (["1:4"], False),
    # Only a subset of the requested residues is present.
    (["3:9"], True),
    # None of the requested residues are present.
    (["9:11"], True),
    # Multiple regions, one fully present and one fully absent.
    (["1:2", "9:11"], True),
    # Chain-specific region that exists in the structure.
    (["A-1:4"], False),
    # Chain-specific region that does not exist in the structure.
    (["C-1:4"], True),
    # Chain-specific region on chain B, which only has resids 1-2.
    (["B-1:4"], True),
])
def test_missing_idr_regions_warn(caplog, id_regions, expected):
    """
    Tests that a warning is logged when -id-regions requests residues
    that are not present in the input structure.

    The structure contains resids 1-4 in chain A and resids 1-2 in chain B.
    """
    system = _make_idr_system()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        AnnotateIDRs(id_regions=id_regions).run_system(system)

    if expected:
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == 'WARNING'
        assert 'missing' in caplog.records[0].getMessage().lower()
    else:
        assert caplog.records == []


@pytest.mark.parametrize('id_regions, expected', [
    # No chain specified: an info message is logged.
    (["1:4"], True),
    # Chain specified: no info message.
    (["A-1:4"], False),
    # Mixed: at least one region without a chain.
    (["A-1:2", "3:4"], True),
])
def test_no_chain_specifier_info(caplog, id_regions, expected):
    """
    Tests that an info message is logged when a region has no chain
    specifier, since it will be applied to all chains.
    """
    system = _make_idr_system()

    caplog.clear()
    with caplog.at_level(logging.INFO):
        AnnotateIDRs(id_regions=id_regions).run_system(system)

    info_records = [rec for rec in caplog.records if rec.levelname == 'INFO']
    if expected:
        assert any('all chains' in rec.getMessage() for rec in info_records)
    else:
        assert not any('all chains' in rec.getMessage() for rec in info_records)
