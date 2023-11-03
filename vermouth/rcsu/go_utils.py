# Copyright 2023 University of Groningen
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
Utilities for Go model processors.
"""
from vermouth.molecule import attributes_match

def get_go_type_from_attributes(molecule, prefix, **kwargs):
    """
    Find all nodes that satisfy a number of attributes specified
    as kwargs and have a specific atomtype prefix.

    Parameters
    ----------
    molecule: :class:vermouth.Molecule
    prefix: str
        the atom-type prefix of the Go virtual side
    kwargs:
        any number of attributes

    Yields
    ------
    str
        the atom-type

    Raises
    ------
    IOError
        If no node can be found that matches attributes
        and prefix an IOError is raised.
    """
    for node in molecule.nodes:
        attrs = molecule.nodes[node]
        if attributes_match(attrs, kwargs) and attrs['atype'].startswith(prefix):
            yield attrs['atype']
    else:
        resid = kwargs['resid']
        chain = kwargs['chain']
        raise ValueError(f"Could not find GoVs with resid {resid} in chain {chain}.")

def _in_resid_region(resid, regions):
    """
    Check if resid falls in regions interval.

    Parameters
    ----------
    resid: int
        the resid of a molecule
    regions: list[tuple(int)]
        a list of the intervals

    Returns
    -------
    bool
    """
    for limits in regions:
        # perhaps someone gives them as reversed
        low, up = sorted(limits)
        if low <= resid <= up:
            return True
    return False

def _get_bead_size(atype):
    if atype.startswith("S"):
        bead_size = "small"
    elif atype.startswith("T"):
        bead_size = "tiny"
    else:
        bead_size = "regular"
    return bead_size
