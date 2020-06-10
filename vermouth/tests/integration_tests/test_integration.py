# Copyright 2020 University of Groningen
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

from .. import datafiles
from pathlib import Path
import pytest
import subprocess

MARTINIZE2 = 'martinize2'  # TODO: turn into pytest cli argument

INTEGRATION_DATA = Path(datafiles.TEST_DATA/'integration_tests')

PATTERN = '{path}/{tier}/{protein}/martinize2/'


def compare_file(file1, file2):
    with open(file1) as f1, open(file2) as f2:
        assert f1.readlines() == f2.readlines()


@pytest.mark.parametrize("tier, protein", [
    ['tier-0', 'mini-protein1_betasheet'],
    ['tier-0', 'mini-protein2_helix'],
    ['tier-0', 'mini-protein3_trp-cage'],
    # ['tier-1', 'bpti'],
    # ['tier-1', 'lysozyme'],
    # ['tier-1', 'villin'],
    # ['tier-2', 'barnase_barstar'],
    # ['tier-2', 'dna'],
    # ['tier-2', 'gpa_dimer'],
])
def test_integration_protein(tmp_path, tier, protein):
    data_path = Path(PATTERN.format(path=INTEGRATION_DATA, tier=tier, protein=protein))

    with open(data_path / 'command') as cmd_file:
        command = cmd_file.read().strip()
    assert command
    command = command.format(inpath=data_path, martinize2=MARTINIZE2)
    command = command.replace('\n', ' ')

    proc = subprocess.Popen(command, cwd=tmp_path)
    exit_code = proc.wait(timeout=60)
    assert exit_code == 0

    for new_file in tmp_path.iterdir():
        filename = new_file.name
        reference_file = data_path/filename
        assert reference_file.is_file()
        compare_file(reference_file, new_file)
