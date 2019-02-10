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

import itertools
import pytest
from hypothesis import given, assume
from hypothesis import strategies as st
from vermouth import System
from vermouth.processors.name_moltype import NameMolType
from .molecule_strategies import random_molecule

@st.composite
def molecules_and_moltypes(draw,
                           min_moltypes=1, max_moltypes=4,
                           min_size=None, max_size=None):
    """
    Generates a list of molecules and the list of the expected moltypes.
    """
    if min_size is None:
        min_size = min_moltypes
    if max_size is not None and max_size < min_size:
        raise ValueError('max_size ({}) must be greater than min_size ({}), or None.'
                         .fornat(max_size, min_size))
    if max_moltypes < min_moltypes:
        raise ValueError('max_moltypes ({}) must be greater than min_moltypes ({}).' 
                         .fornat(max_moltypes, min_moltypes))
                         
    # We first generate the list of expected moltypes, we can then use it as
    # keys to draw the molecules with shared strategies. The moltypes must
    # match what NameMolType would generate, so they must have the form
    # molecule_0, molecule_1, molecule_x, where the index is increasing based
    # on the first occurence of the molecule type.
    # pre_moltypes will be the moltype names before we rename them based on
    # their order.
    n_moltypes = draw(st.integers(min_value=min_moltypes, max_value=max_moltypes))
    unique_pre_moltypes = range(n_moltypes)
    pre_moltypes_sampled = draw(st.lists(st.sampled_from(unique_pre_moltypes),
                                         min_size=min_size, max_size=max_size))
    if not pre_moltypes_sampled:
        return [], []

    moltype_index = 0
    pre_to_moltype = {pre_moltypes_sampled[0]: 'molecule_{}'.format(moltype_index)}
    moltypes = []
    for pre_moltype in pre_moltypes_sampled:
        if pre_moltype not in pre_to_moltype:
            moltype_index += 1
            pre_to_moltype[pre_moltype] = 'molecule_{}'.format(moltype_index)
        moltypes.append(pre_to_moltype[pre_moltype])

    molecule_share_template = tuple(st.shared(random_molecule(), key=moltype)
                                    for moltype in moltypes)
    molecules = draw(st.tuples(*molecule_share_template))

    # So far we made sure that molecules with a different moltype are drawn
    # separately. Though, two different draws may result in equal molecules,
    # especially as molecules are reduced whn shrinking the draws.
    representatives = [
        next(group)[0]
        for _, group in itertools.groupby(
                zip(molecules, moltypes), key=lambda x: x[1])
    ]
    assume(all(not mol1.share_moltype_with(mol2)
               for mol1, mol2 in itertools.combinations(representatives, 2)))

    # Each molecule must be its own instance as molecules are modified in place.
    molecules = [molecule.copy() for molecule in molecules]

    return [molecules, moltypes]


@pytest.mark.parametrize('deduplicate', (True, False))
@given(mols_and_moltypes=molecules_and_moltypes())
def test_name_moltype(mols_and_moltypes, deduplicate):
    molecules, moltypes = mols_and_moltypes

    system = System()
    system.molecules = molecules
    
    if not deduplicate:
        moltypes = ['molecule_{}'.format(i) for i in range(len(molecules))]

    processor = NameMolType(deduplicate=deduplicate)
    processor.run_system(system)

    found_moltypes = [molecule.meta['moltype'] for molecule in system.molecules]
    assert found_moltypes == moltypes
