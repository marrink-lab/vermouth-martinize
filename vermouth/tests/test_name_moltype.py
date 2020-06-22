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
Test for the NameMolType processor.
"""

# Pylint complains about hypothesis strategies not receiving a value for the
# `draw` parameter. This is because the `draw` parameter is implicitly filled
# by hypothesis. The pylint warning is disabled for the file instead of being
# disabled at every call of a strategy.
# pylint: disable=no-value-for-parameter

from collections import defaultdict
import itertools
from glob import glob
import os.path
import subprocess
import sys
import pytest
from hypothesis import given, assume
from hypothesis import strategies as st
from vermouth import System
from vermouth.processors.name_moltype import NameMolType
from .molecule_strategies import random_molecule
from .datafiles import PDB_HB
from .helper_functions import find_in_path

@st.composite
def molecules_and_moltypes(draw, max_moltypes=4, min_size=0, max_size=None,
                           molecule_kwargs=None):
    """
    Generates a list of molecules and the list of the expected moltypes.

    Parameters
    ----------
    draw:
        Internal for hypothesis.
    max_moltypes: int
        Maximum number of different moltypes.
    min_size: int
        Minimum number of molecules.
    max_size: int, optional
        Maximum number of molecules. If `None`, behaves the same way as
        :func:`hypothesis.strategies.lists`.
    molecule_kwargs: dict, optional
        Arguments to pass to :func:`random_molecule`.

    Returns
    -------
    hypothesis.searchstrategy.lazy.LazyStrategy
        A list of molecules and a list of moltype names corresponding to these
        molecules as they are expected to be assigned by
        :class:`vermouth.processors.name_moltype.NameMolType`.
    """
    if max_size is not None and max_size < min_size:
        raise ValueError('max_size ({}) must be greater than min_size ({}), or None.'
                         .format(max_size, min_size))
    if molecule_kwargs is None:
        molecule_kwargs = {}

    # We first generate the list of expected moltypes, we can then use it as
    # keys to draw the molecules with shared strategies. The moltypes must
    # match what NameMolType would generate, so they must have the form
    # molecule_0, molecule_1, molecule_x, where the index is increasing based
    # on the first occurence of the molecule type.
    # pre_moltypes will be the moltype names before we rename them based on
    # their order.
    n_moltypes = draw(st.integers(min_value=1, max_value=max_moltypes))
    unique_pre_moltypes = range(n_moltypes)
    pre_moltypes_sampled = draw(st.lists(st.sampled_from(unique_pre_moltypes),
                                         min_size=min_size, max_size=max_size))
    if not pre_moltypes_sampled:
        return [], []

    moltype_index = -1  # Will be 0 for the first moltype
    pre_to_moltype = {}
    moltypes = []
    for pre_moltype in pre_moltypes_sampled:
        if pre_moltype not in pre_to_moltype:
            moltype_index += 1
            pre_to_moltype[pre_moltype] = 'molecule_{}'.format(moltype_index)
        moltypes.append(pre_to_moltype[pre_moltype])

    sharing = defaultdict(lambda: draw(random_molecule(**molecule_kwargs)))
    molecules = [sharing[moltype] for moltype in moltypes]

    # So far we made sure that molecules with a different moltype are drawn
    # separately. Though, two different draws may result in equal molecules,
    # especially as molecules are reduced when shrinking the draws.
    representatives = [
        next(group)[0]
        for _, group in itertools.groupby(
            zip(molecules, moltypes), key=lambda x: x[1]
        )
    ]
    assume(all(not mol1.share_moltype_with(mol2)
               for mol1, mol2 in itertools.combinations(representatives, 2)))

    # Each molecule must be its own instance as molecules are modified in place.
    molecules = [molecule.copy() for molecule in molecules]

    return [molecules, moltypes]


@pytest.mark.parametrize('deduplicate', (True, False))
@given(mols_and_moltypes=molecules_and_moltypes(
    max_size=4, max_moltypes=3, molecule_kwargs={'max_nodes': 3, 'max_meta': 3},
))
def test_name_moltype(mols_and_moltypes, deduplicate):
    """
    The NameMolType processor works as expected with and withour deduplication.
    """
    molecules, moltypes = mols_and_moltypes

    system = System()
    system.molecules = molecules

    if not deduplicate:
        moltypes = ['molecule_{}'.format(i) for i in range(len(molecules))]

    processor = NameMolType(deduplicate=deduplicate)
    processor.run_system(system)

    found_moltypes = [molecule.meta['moltype'] for molecule in system.molecules]
    assert found_moltypes == moltypes


@pytest.mark.parametrize('deduplicate', (True, False))
def test_martinize2_moltypes(tmpdir, deduplicate):
    """
    Run martinize2 and make sure the ITP file produced have the expected names.
    """
    martinize2 = find_in_path()

    command = [
        sys.executable,
        martinize2,
        '-f', str(PDB_HB),
        '-o', 'topol.top',
        '-x', 'out.pdb',
        '-ignore', 'HOH', '-ignore', 'HEME',
    ]
    if deduplicate:
        expected = ['molecule_{}.itp'.format(i) for i in range(2)]
        n_outputs = 2
    else:
        command.append('-sep')
        n_outputs = 4
    expected = ['molecule_{}.itp'.format(i) for i in range(n_outputs)]

    proc = subprocess.run(command, cwd=str(tmpdir), timeout=90, check=False)
    assert proc.returncode == 0

    itp_files = sorted(os.path.basename(fname) for fname in glob(str(tmpdir / '*.itp')))
    assert itp_files == expected
