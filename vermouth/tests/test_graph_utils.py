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
# limitations under the License.import pytest

import pytest
import vermouth


def basic_molecule(node_data):
    mol = vermouth.Molecule()
    for idx, node in enumerate(node_data):
        mol.add_node(idx, **node)
    return mol


@pytest.mark.parametrize('node_data_in,expected_node_data', [
        ([], []),
        ([{'atomname': 'H3'}], [{'atomname': 'H3', 'element': 'H'}]),
        ([{'atomname': '1H3'}], [{'atomname': '1H3', 'element': 'H'}]),
        ([{'atomname': 'H3'}, {'atomname': '1H3'}], [{'atomname': 'H3', 'element': 'H'}, {'atomname': '1H3', 'element': 'H'}]),
        ([{'atomname': 'Cl1', 'element': 'Cl', 'attr': None}, {'atomname': '31C3'}], [{'atomname': 'Cl1', 'element': 'Cl', 'attr': None}, {'atomname': '31C3', 'element': 'C'}]),
        ([{'element': 'Cl'}, {'atomname': '31C3'}], [{'element': 'Cl'}, {'atomname': '31C3', 'element': 'C'}]),
        ]
        )
def test_add_element_attr(node_data_in, expected_node_data):
    mol = basic_molecule(node_data_in)
    vermouth.graph_utils.add_element_attr(mol)
    expected = basic_molecule(expected_node_data)
    assert mol.nodes(data=True) == expected.nodes(data=True)


@pytest.mark.parametrize('node_data_in,exception', [
        ([{'atomname': '1234'}], ValueError),
        ([{'peanuts': '1H3'}], ValueError),
        ([{'atomname': 'H3'}, {'atomname': '1234'}], ValueError),
        ([{'atomname': 'H3'}, {'peanuts': '1234'}], ValueError),
        ]
        )
def test_add_element_attr_errors(node_data_in, exception):
    mol = basic_molecule(node_data_in)
    with pytest.raises(exception):
        vermouth.graph_utils.add_element_attr(mol)
