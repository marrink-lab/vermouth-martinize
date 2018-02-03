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
# limitations under the License.import os

from pathlib import Path

# TODO: Make that cleaner with pkg_resources
HERE = Path(__file__).parent
TEST_DATA = HERE / 'data'

# PDB files with a single molecule
PDB_PROTEIN = TEST_DATA / '1bta.pdb'
PDB_PARTIALLY_PROTEIN = TEST_DATA / '1bta_mutated.pdb'  # LYS replaced by UNK
PDB_NOT_PROTEIN = TEST_DATA / 'heme.pdb'

DSSP_OUTPUT = TEST_DATA / 'dssp_1bta.ssd'


