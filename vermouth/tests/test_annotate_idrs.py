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
import pytest
from vermouth.processors.annotate_idrs import AnnotateIDRs
from .datafiles import PDB_ALA5_CG
from vermouth.pdb.pdb import read_pdb
import networkx as nx

def protein():
    """
    Read a PDB file describing a protein at CG resolution.
    Embed the content of the PDB in a :class:`vermouth.system.System`.
    """
    molecules = read_pdb(PDB_ALA5_CG)
    assert len(molecules) == 1
    molecule = molecules[0]
    molecule.remove_edges_from(list(molecule.edges))
    _resids = nx.get_node_attributes(molecule, "resid")
    nx.set_node_attributes(molecule, _resids, "_old_resid")
    return molecule

@pytest.mark.parametrize('idr_regions, expected', [
    (
    [(1,3)],
    [True, True, True, True, True, True, False, False, False, False]
    ),
    (
    [(1,2),(4,5)],
    [True, True, True, True, False, False, True, True, True, True]
    ),
    (
    [],
    [False, False, False, False, False, False, False, False, False, False]
    )
])
def test_make_disorder_string(idr_regions, expected):
    molecule = protein()
    AnnotateIDRs(idr_regions=idr_regions).run_molecule(molecule)
    result = []
    for key, node in molecule.nodes.items():
        if molecule.nodes[key].get("cgidr"):
            result.append(True)
        else:
            result.append(False)
    assert result == expected

