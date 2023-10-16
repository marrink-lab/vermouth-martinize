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
Test the writing of TOP file.
"""
import pytest
import vermouth
from vermouth.file_writer import DeferredFileWriter
from vermouth.gmx.topology import (convert_sigma_epsilon,
                                   write_atomtypes,
                                   write_nonbond_params,
                                   write_gmx_topology,
                                   Atomtype,
                                   NonbondParam)
from vermouth.tests.gmx.test_itp import dummy_molecule

@pytest.mark.parametrize('atomtypes, expected',
        # simple atomtype
        #'Atomtype', 'molecule node sigma epsilon meta
        ((
        [{"node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 \n"]),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {}},
         {"node":1, "sigma": 0.47, "epsilon": 2.5, "meta": {}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 \n",
         "B 72 0 A 0.47000000 2.50000000 \n"]),
        (
        [{"node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {"comment": ["comment"]}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 ;comment\n"]),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {"group": "g1"}},
         {"node":1, "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"node":2, "sigma": 0.47, "epsilon": 2.5, "meta": {"group": "g1"}}],
        ['[ atomtypes ]\n', 
         'B 72 0 A 0.44000000 3.30000000 \n', 
         '; g1\n', 
         'A 72 0 A 0.43000000 2.30000000 \n', 
         'C 72 0 A 0.47000000 2.50000000 \n']
        ),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {"ifdef": "g1"}},
         {"node":1, "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"node":2, "sigma": 0.47, "epsilon": 2.5, "meta": {"ifdef": "g1"}}],
        ['[ atomtypes ]\n', 
         'B 72 0 A 0.44000000 3.30000000 \n', 
         '#ifdef g1\n', 
         'A 72 0 A 0.43000000 2.30000000 \n', 
         'C 72 0 A 0.47000000 2.50000000 \n',
         '#endif']
        ),
        ))
def test_no_header(tmp_path, dummy_molecule, atomtypes, expected):
    """
    Test that the atomtypes directive is properly written.
    """
    dummy_sys = vermouth.system.System("")
    for atype in atomtypes:
        # somewhat hacky workaround because pytest doesn't
        # allow the use of fixtures in parametrize
        atype['molecule'] = dummy_molecule
        dummy_sys.gmx_topology_params['atomtypes'].append(Atomtype(**atype))

    outpath = tmp_path / 'out.itp'
    write_atomtypes(dummy_sys, outpath)
    DeferredFileWriter().write()

    with open(str(outpath)) as infile:
        for line, ref_line in zip(infile.readlines(), expected):
            assert line == ref_line
