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

import collections
import pytest
import networkx as nx

import vermouth.graphing.grappa as grappa

# pylint: disable=redefined-outer-name



RefEntry = collections.namedtuple(
    'RefEntry', 'string nodes edges attributes'
)

REFERENCE = {
    'BB': RefEntry(
        'N(H,.) CA(HA,.) C(O1,.) @CA {chiral:(N,C,HA)}',
        ['N', 'HA', 'C', 'H', 'O1', 'CA'],
        [('N', 'CA'), ('N', 'H'), ('HA', 'CA'), ('C', 'CA'), ('C', 'O1')],
        {'CA': {'chiral': '(N,C,HA)', 'stub': 1}, 'C': {'stub': 1},
         'N': {'stub': 1}},
    ),
    'GLY': RefEntry(
        '<BB> @HA =HA1 @CA HA2',
        ['N', 'HA1', 'C', 'H', 'HA2', 'O1', 'CA'],
        [('N', 'CA'), ('N', 'H'), ('HA1', 'CA'), ('C', 'CA'),
         ('C', 'O1'), ('HA2', 'CA')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'ALA': RefEntry(
        '<BB> @CA CB(HB[1-3])',
        ['N', 'CB', 'HA', 'C', 'H', 'HB2', 'HB3', 'HB1', 'O1', 'CA'],
        [('N', 'CA'), ('N', 'H'), ('CB', 'HB2'), ('CB', 'HB3'), ('CB', 'CA'),
         ('CB', 'HB1'), ('HA', 'CA'), ('C', 'CA'), ('C', 'O1')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'ASP': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(OD1,OD2)',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG',
         'OD1', 'OD2'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'OD1'), ('CG', 'OD2')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'ASN': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(OD1) ND2(HD21,HD22)',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG',
         'OD1', 'ND2', 'HD21', 'HD22'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'OD1'), ('CG', 'ND2'), ('ND2', 'HD21'), ('ND2', 'HD22')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'SER': RefEntry(
        '<BB> @CA CB(HB1,HB2) OG HG',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'OG', 'HG'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'OG'),
         ('OG', 'HG')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'CYS': RefEntry(
        '<BB> @CA CB(HB1,HB2) SG HG',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'SG', 'HG'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'SG'),
         ('SG', 'HG')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'MET': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(HD1,HD2) SD CE(HE[1-3])',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'HG1',
         'HG2', 'CD', 'HD1', 'HD2', 'SD', 'CE', 'HE1', 'HE2', 'HE3'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'HG1'), ('CG', 'HG2'), ('CG', 'CD'), ('CD', 'HD1'),
         ('CD', 'HD2'), ('CD', 'SD'), ('SD', 'CE'), ('CE', 'HE1'),
         ('CE', 'HE2'), ('CE', 'HE3')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'THR': RefEntry(
        '<BB> @CA CB (OG1 HG1) CG2 (HG2[1-3])',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'OG1',
         'HG1', 'CG2', 'HG21', 'HG22', 'HG23'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'OG1'), ('CB', 'CG2'), ('OG1', 'HG1'),
         ('CG2', 'HG21'), ('CG2', 'HG22'), ('CG2', 'HG23')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'VAL': RefEntry(
        '<BB> @CA CB(HB,CG1(HG1[1-3]),CG2(HG2[1-3]))',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB', 'CG1', 'HG11', 'HG12',
         'HG13', 'CG2', 'HG21', 'HG22', 'HG23'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB'), ('CB', 'CG1'), ('CB', 'CG2'),
         ('CG1', 'HG11'), ('CG1', 'HG12'), ('CG1', 'HG13'), ('CG2', 'HG21'),
         ('CG2', 'HG22'), ('CG2', 'HG23')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'ILE': RefEntry(
        '<BB> @CA CB(HB,CG2(HG2[1-3])) CG1(HG1[1-2]) CD(HD[1-3])',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB', 'CG2', 'HG21', 'HG22',
         'HG23', 'CG1', 'HG11', 'HG12', 'CD', 'HD1', 'HD2', 'HD3'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB'), ('CB', 'CG2'), ('CB', 'CG1'),
         ('CG2', 'HG21'), ('CG2', 'HG22'), ('CG2', 'HG23'), ('CG1', 'HG11'),
         ('CG1', 'HG12'), ('CG1', 'CD'), ('CD', 'HD1'), ('CD', 'HD2'),
         ('CD', 'HD3')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'LEU': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(HG1,CD1(HD1[1-3]),CD2(HD2[1-3]))',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'HG1',
         'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22', 'HD23'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'HG1'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'HD11'),
         ('CD1', 'HD12'), ('CD1', 'HD13'), ('CD2', 'HD21'), ('CD2', 'HD22'),
         ('CD2', 'HD23')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'GLU': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(OE1,OE2)',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'HG1',
         'HG2', 'CD', 'OE1', 'OE2'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'HG1'), ('CG', 'HG2'), ('CG', 'CD'), ('CD', 'OE1'),
         ('CD', 'OE2')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'GLN': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(OE1) NE2(HE21,HE22)',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'HG1',
         'HG2', 'CD', 'OE1', 'NE2', 'HE21', 'HE22'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'HG1'), ('CG', 'HG2'), ('CG', 'CD'), ('CD', 'OE1'),
         ('CD', 'NE2'), ('NE2', 'HE21'), ('NE2', 'HE22')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'PRO': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(HD1,HD2) !C',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'HG1',
         'HG2', 'CD', 'HD1', 'HD2'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('C', 'CD'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'HG1'), ('CG', 'HG2'), ('CG', 'CD'), ('CD', 'HD1'),
         ('CD', 'HD2')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'HIS': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG CE1(HE1) ND1 CE1(HE1) NE2(HE2 {pKa:6.04}) CD2(HD2) !CG',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'CE1',
         'HE1', 'ND1', 'NE2', 'HE2', 'CD2', 'HD2'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'CE1'), ('CG', 'CD2'), ('CE1', 'HE1'), ('CE1', 'ND1'),
         ('CE1', 'NE2'), ('NE2', 'HE2'), ('HE2', 'CD2'), ('CD2', 'HD2')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1},
         'N': {'stub': 1}, 'HE2': {'pKa': '6.04'}},
    ),
    'PHE': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG CD1(HD1) CE1(HE1) CZ(HZ) CE2(HE2) CD2(HD2) !CG',
        ['HZ', 'CE2', 'C', 'N', 'HD2', 'CD2', 'HD1', 'CD1', 'HB1', 'HE2',
         'O1', 'HB2', 'CB', 'HA', 'CG', 'H', 'CE1', 'HE1', 'CZ', 'CA'],
        [('HZ', 'CZ'), ('CE2', 'HE2'), ('CE2', 'CZ'), ('CE2', 'CD2'),
         ('C', 'CA'), ('C', 'O1'), ('N', 'CA'), ('N', 'H'), ('HD2', 'CD2'),
         ('HD1', 'CD1'), ('CD1', 'CE1'), ('CD1', 'CG'), ('HB1', 'CB'),
         ('HB2', 'CB'), ('CB', 'CA'), ('CB', 'CG'), ('HA', 'CA'),
         ('CE1', 'CZ'), ('CE1', 'HE1'), ('CD2', 'CG')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'TYR': RefEntry(
        '<PHE> -HZ @CZ OH HH {pKa:10.10}',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'CD1',
         'HD1', 'CE1', 'HE1', 'CZ', 'CE2', 'HE2', 'CD2', 'HD2', 'OH', 'HH'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'CD1'), ('CD1', 'HD1'), ('CD1', 'CE1'), ('CE1', 'HE1'),
         ('CE1', 'CZ'), ('CZ', 'CE2'), ('CZ', 'OH'), ('CE2', 'HE2'),
         ('CE2', 'CD2'), ('CD2', 'HD2'), ('OH', 'HH'), ('CD2', 'CG')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1},
         'N': {'stub': 1}, 'HH': {'pKa': '10.10'}},
    ),
    'LYS': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(HD1,HD2) CE(HE1,HE2) NZ(HZ1,HZ2,HZ3)',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'HG1',
         'HG2', 'CD', 'HD1', 'HD2', 'CE', 'HE1', 'HE2', 'NZ', 'HZ1', 'HZ2',
         'HZ3'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'HG1'), ('CG', 'HG2'), ('CG', 'CD'), ('CD', 'HD1'),
         ('CD', 'HD2'), ('CD', 'CE'), ('CE', 'HE1'), ('CE', 'HE2'),
         ('CE', 'NZ'), ('NZ', 'HZ1'), ('NZ', 'HZ2'), ('NZ', 'HZ3')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
    'ARG': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG(HG1,HG2) CD(HD1,HD2) NE(HE) CZ(NH1(HH11,HH12),NH2(HH21,HH22 {pKa:12.10}))',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'HG1',
         'HG2', 'CD', 'HD1', 'HD2', 'NE', 'HE', 'CZ', 'NH1', 'HH11', 'HH12',
         'NH2', 'HH21', 'HH22'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'HG1'), ('CG', 'HG2'), ('CG', 'CD'), ('CD', 'HD1'),
         ('CD', 'HD2'), ('CD', 'NE'), ('NE', 'HE'), ('NE', 'CZ'),
         ('CZ', 'NH1'), ('CZ', 'NH2'), ('NH1', 'HH11'), ('NH1', 'HH12'),
         ('NH2', 'HH21'), ('NH2', 'HH22')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1},
         'N': {'stub': 1}, 'HH22': {'pKa': '12.10'}},
    ),
    'TRP': RefEntry(
        '<BB> @CA CB(HB1,HB2) CG CD1(HD1) NE1(HE1) CE2 CZ2(HZ2) CH2(HH2) CZ3(HZ3) CE3(HE3) CD2 !CG @CD2 !CE2',
        ['N', 'H', 'CA', 'HA', 'C', 'O1', 'CB', 'HB1', 'HB2', 'CG', 'CD1',
         'HD1', 'NE1', 'HE1', 'CE2', 'CZ2', 'HZ2', 'CH2', 'HH2', 'CZ3', 'HZ3',
         'CE3', 'HE3', 'CD2'],
        [('N', 'H'), ('N', 'CA'), ('CA', 'HA'), ('CA', 'C'), ('CA', 'CB'),
         ('C', 'O1'), ('CB', 'HB1'), ('CB', 'HB2'), ('CB', 'CG'),
         ('CG', 'CD1'), ('CD1', 'HD1'), ('CD1', 'NE1'), ('NE1', 'HE1'),
         ('NE1', 'CE2'), ('CE2', 'CZ2'), ('CZ2', 'HZ2'), ('CZ2', 'CH2'),
         ('CH2', 'HH2'), ('CH2', 'CZ3'), ('CZ3', 'HZ3'), ('CZ3', 'CE3'),
         ('CE3', 'HE3'), ('CE3', 'CD2'), ('CD2', 'CG'), ('CD2', 'CE2')],
        {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}},
    ),
}


@pytest.fixture()
def graph_dict():
    names_to_use = ['BB', 'PHE']
    attributes = {'CA': {'chiral': '(N,C,HA)'}, 'C': {'stub': 1}, 'N': {'stub': 1}}
    graphs = {}
    for name in names_to_use:
        graphs[name] = nx.Graph()
        graphs[name].add_nodes_from(attributes.items())
        graphs[name].add_edges_from(REFERENCE[name].edges)
    #graphs['BB'].nodes['CA']['stub'] = 1
    return graphs


@pytest.mark.parametrize('grappa_string, ref_string', (
    (
        '/#=1-5/C#(O#,H#1,H#2)/ @C5 H53 @O4 -H41 !C1',
        'C1(O1,H11,H12) C2(O2,H21,H22) C3(O3,H31,H32) C4(O4,H41,H42) C5(O5,H51,H52) @C5 H53 @O4 -H41 !C1',
    ),
    (
        '/#=1-5/C#(O# H#1,H#2)/ @C5 H53 @O4 -H41 !C1',
        'C1(O1 H11,H12) C2(O2 H21,H22) C3(O3 H31,H32) C4(O4 H41,H42) C5(O5 H51,H52) @C5 H53 @O4 -H41 !C1',
    ),
))
def test_preprocess(grappa_string, ref_string):
    assert grappa.preprocess(grappa_string) == ref_string


@pytest.mark.parametrize('name, grappa_string, nodes, edges', (
    (name, ref.string, ref.nodes, ref.edges)
    for name, ref in REFERENCE.items()
))
def test_process_graph(name, grappa_string, nodes, edges, graph_dict):
    residue = grappa.process(grappa_string, graphs=graph_dict)

    # The residues are undirected graphs. When comparing their edges, neither
    # the order of the edges themselves, nor the order of the nodes within an
    # edge should matter. We sort the edges inside and out.
    residue_edges = sorted([sorted(edge) for edge in residue.edges])
    reference_edges = sorted([sorted(edge) for edge in edges])
    assert residue_edges == reference_edges
    assert sorted(residue.nodes) == sorted(nodes)


@pytest.mark.parametrize('name, grappa_string, attributes', (
    (name, ref.string, ref.attributes)
    for name, ref in REFERENCE.items()
))
def test_process_attributes(name, grappa_string, attributes, graph_dict):
    residue = grappa.process(grappa_string, graphs=graph_dict)

    # Make sure that the expected attributes are created.
    for atom, atom_attributes in attributes.items():
        for key, value in atom_attributes.items():
            assert key in residue.nodes.get(atom, {})
            assert value == residue.nodes[atom][key]

    # Make sure there are no unexpected attributes.
    for node, atom_attributes in residue.nodes.items():
        for key, value in atom_attributes.items():
            assert key in attributes.get(node, {})
            # If the attribute exist for the atom, we checked already that the
            # value was as expected.
