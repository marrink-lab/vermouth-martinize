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
Unittests for the CIF reader.
"""

import numpy as np
import pytest
import logging
import sys
from CifFile import ReadCif
import os.path
import subprocess

from vermouth.forcefield import ForceField
from vermouth.gmx import read_itp
import vermouth.pdb.cif as cif
from vermouth.tests.datafiles import (CIF_PROTEIN, CIF_NO_CELL, CIF_PDB_COMPARE, CIF_MULTI, CIF_MISSING_RESNAME,
                                      CIF_MISSING_ATOMNAME)
from ..helper_functions import find_in_path
from ..integration_tests.test_integration import compare_pdb, assert_equal_blocks


@pytest.mark.parametrize('input, expected',
                         (
                                 (CIF_PROTEIN, np.array([50.840, 42.770, 28.950, 90, 90, 90])),
                                 (CIF_NO_CELL, np.array([1, 1, 1, 90, 90, 90])),

                         )
                         )
def test_cell(input, expected):
    input_data = ReadCif(str(input))
    model_name = '1UBQ'

    assert cif._cell(input_data, model_name) == pytest.approx(expected)


def cif_lines():
    """
    lines from CIF_PROTEIN of the first two residues
    """
    cif = '''ATOM   1   N N   . MET A 1 1  ? 27.340 24.430 2.614  1.00 9.67  ? 1   MET A N   1
ATOM   2   C CA  . MET A 1 1  ? 26.266 25.413 2.842  1.00 10.38 ? 1   MET A CA  1
ATOM   3   C C   . MET A 1 1  ? 26.913 26.639 3.531  1.00 9.62  ? 1   MET A C   1
ATOM   4   O O   . MET A 1 1  ? 27.886 26.463 4.263  1.00 9.62  ? 1   MET A O   1
ATOM   5   C CB  . MET A 1 1  ? 25.112 24.880 3.649  1.00 13.77 ? 1   MET A CB  1
ATOM   6   C CG  . MET A 1 1  ? 25.353 24.860 5.134  1.00 16.29 ? 1   MET A CG  1
ATOM   7   S SD  . MET A 1 1  ? 23.930 23.959 5.904  1.00 17.17 ? 1   MET A SD  1
ATOM   8   C CE  . MET A 1 1  ? 24.447 23.984 7.620  1.00 16.11 ? 1   MET A CE  1
ATOM   9   N N   . GLN A 1 2  ? 26.335 27.770 3.258  1.00 9.27  ? 2   GLN A N   1
ATOM   10  C CA  . GLN A 1 2  ? 26.850 29.021 3.898  1.00 9.07  ? 2   GLN A CA  1
ATOM   11  C C   . GLN A 1 2  ? 26.100 29.253 5.202  1.00 8.72  ? 2   GLN A C   1
ATOM   12  O O   . GLN A 1 2  ? 24.865 29.024 5.330  1.00 8.22  ? 2   GLN A O   1
ATOM   13  C CB  . GLN A 1 2  ? 26.733 30.148 2.905  1.00 14.46 ? 2   GLN A CB  1
ATOM   14  C CG  . GLN A 1 2  ? 26.882 31.546 3.409  1.00 17.01 ? 2   GLN A CG  1
ATOM   15  C CD  . GLN A 1 2  ? 26.786 32.562 2.270  1.00 20.10 ? 2   GLN A CD  1
ATOM   16  O OE1 . GLN A 1 2  ? 27.783 33.160 1.870  1.00 21.89 ? 2   GLN A OE1 1
ATOM   17  N NE2 . GLN A 1 2  ? 25.562 32.733 1.806  1.00 19.49 ? 2   GLN A NE2 1
'''

    data = []
    for line in cif.splitlines():
        tokens = line.split()
        anum = tokens[1]
        aname = tokens[-2]
        resname = tokens[-4]
        chain = tokens[-3]
        resid = tokens[-5]
        insert_code = tokens[9]
        x = tokens[10]
        y = tokens[11]
        z = tokens[12]
        position = np.array([tokens[10], tokens[11], tokens[12]], dtype=float) / 10
        element = tokens[2]
        charge = tokens[-6]
        occupancy = tokens[13]
        temp_factor = tokens[14]
        model = tokens[-1]
        data.append({"atomid": int(anum),
                     "atomname": str(aname),
                     "resname": str(resname),
                     "chain": str(chain),
                     "x": float(x),
                     "y": float(y),
                     "z": float(z),
                     "resid": int(resid),
                     "position": position,
                     "element": str(element),
                     "charge": None,
                     "insertion_code": str(insert_code),
                     "occupancy": float(occupancy),
                     "temp_factor": float(temp_factor),
                     "model": int(model)})

    return data


def test_read_cif_file():
    molecule = cif.read_cif_file(CIF_PROTEIN)[0]

    reference_data = cif_lines()

    for i in sorted(molecule.nodes)[:len(reference_data)]:
        assert all(j in molecule.nodes[i].keys() for j in reference_data[i].keys())

        for key, value in molecule.nodes[i].items():
            assert value == pytest.approx(reference_data[i][key])


def test_CIFReader():
    processor = cif.CIFReader(file=CIF_PROTEIN, exclude=('HOH'), ignh=False)

    molecules = processor.reader()

    assert len(molecules) == 1


def test_equal_output(tmp_path):
    martinize2 = find_in_path()

    command0 = [
        sys.executable,
        martinize2,
        '-f', str(CIF_PROTEIN),
        '-o', 'cif.top',
        '-x', 'cif.pdb',
        '-ignore', 'HOH', '-name', 'cif', '-ss', 'C'
    ]
    command1 = [
        sys.executable,
        martinize2,
        '-f', str(CIF_PDB_COMPARE),
        '-o', 'pdb.top',
        '-x', 'pdb.pdb',
        '-ignore', 'HOH', '-name', 'pdb', '-maxwarn', 'missing-flag:1'
    ]
    # run commands
    proc0 = subprocess.run(command0, cwd=tmp_path, timeout=90, check=False)
    assert proc0.returncode == 0
    proc1 = subprocess.run(command1, cwd=tmp_path, timeout=90, check=False)
    assert proc1.returncode == 0

    # find files and assert only 2 files of each
    itp_files = sorted(os.path.abspath(fname) for fname in tmp_path.glob('*.itp'))
    pdb_files = sorted(os.path.abspath(fname) for fname in tmp_path.glob('*.pdb'))
    assert len(itp_files) == 2
    assert len(pdb_files) == 2

    # compare pdb. easily read in
    compare_pdb(pdb_files[0], pdb_files[1])

    # need to take care with itps because we have to use different names to make sure we don't overwrite in tmp_path.
    cif_itp = [file for file in itp_files if 'cif' in file][0]
    pdb_itp = [file for file in itp_files if 'pdb' in file][0]

    dummy_ff = ForceField(name='dummy')
    with open(cif_itp) as fn1:
        read_itp(fn1, dummy_ff)
    dummy_ff2 = ForceField(name='dummy')
    with open(pdb_itp) as fn2:
        read_itp(fn2, dummy_ff2)

    assert_equal_blocks(dummy_ff.blocks['cif_0'],
                        dummy_ff2.blocks['pdb_0'],
                        blocknames_equal=False)

def test_missing_resnames(caplog):
    cif.read_cif_file(CIF_MISSING_RESNAME)
    assert any([rec.levelname == 'WARNING' for rec in caplog.records])


def test_missing_atomnames(caplog):
    cif.read_cif_file(CIF_MISSING_ATOMNAME)
    assert any([rec.levelname == 'WARNING' for rec in caplog.records])


def test_multiple_entries(caplog):
    caplog.set_level(logging.INFO)
    cif.read_cif_file(CIF_MULTI)
    assert any([rec.levelname == 'INFO' for rec in caplog.records])
