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

from pathlib import Path

try:
    import pkg_resources
except ImportError:
    import os
    TEST_DATA = os.path.join(os.path.dirname(__file__), 'data')
    del os
else:
    TEST_DATA = Path(pkg_resources.resource_filename('vermouth.tests', 'data'))
    del pkg_resources

# PDB files with a single molecule
PDB_PROTEIN = TEST_DATA / '1bta.pdb'
PDB_PARTIALLY_PROTEIN = TEST_DATA / '1bta_mutated.pdb'  # LYS replaced by UNK
PDB_NOT_PROTEIN = TEST_DATA / 'heme.pdb'
PDB_ALA5 = TEST_DATA / 'ala5.pdb'
PDB_ALA5_CG = TEST_DATA / 'ala5_cg.pdb'
PDB_TRI_ALANINE = TEST_DATA / 'tri_alanine.pdb'

# Full PDB files
PDB_CYS = TEST_DATA / '2QWO.pdb'  # Contains cystein bridges
PDB_HB = TEST_DATA / '2dn2.pdb'  # Hemoglobin with heme and 2*2 protein chains
PDB_MULTIMODEL = TEST_DATA / '6E8W.pdb'  # HIV-1, 15 models

# DNA
SHORT_DNA = TEST_DATA / 'dna-short.pdb'

DSSP_OUTPUT = TEST_DATA / 'dssp_1bta.ssd'

# Test force fields
FF_UNIVERSAL_TEST = TEST_DATA / 'force_fields' / 'universal-test'
FF_PEPPLANE = TEST_DATA / 'force_fields' / 'pepplane'
FF_MARTINI_TEST = TEST_DATA / 'force_fields' / 'martini-test'

# Mappings
MAP_UNIVERSAL_TEST_PEPPLANE = TEST_DATA / 'mappings' / 'universal-test'


# Clean the namespace so only the data file variables can be imported.
# An other option would be to define __all__, but it is easy to forget to add
# a variable in that list.
del Path
