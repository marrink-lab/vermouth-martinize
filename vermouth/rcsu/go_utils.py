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

def get_go_type_from_attributes(molecule, prefix, method, **kwargs):
    """
    Find all nodes that satisfy a number of attributes specified
    as kwargs and have a specific atomtype prefix.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
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
    KeyError
        If no node can be found that matches attributes
        and prefix an KeyError is raised.
    """
    all_virt_sites = []
    for node in molecule.nodes:
        attrs = molecule.nodes[node]
        if attributes_match(attrs, kwargs) and attrs['atype'].startswith(prefix):
            if method == 0:
                all_virt_sites.append(attrs['atype'])
                break
            if method == 1:
                all_virt_sites.append(attrs['atype'])
                if attrs['atype'][-1] == 'b' or attrs['atype'][-1] == 'd':
                    all_virt_sites.append(attrs['node_id'])

    if not all_virt_sites:
        resid = kwargs['resid']
        chain = kwargs['chain']
        raise KeyError(f"Could not find GoVs with resid {resid} in chain {chain}.")
    
    return all_virt_sites
    

def _in_resid_region(resid, regions):
    """
    Check if resid falls in regions interval.

    Parameters
    ----------
    resid: int
        the resid of a molecule
    regions: list[tuple(int, int)]
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

def _in_chain_and_resid_region(region, resid, chain):
    """
    Check if a chain and resid match

    Parameters
    ----------
    region: dict
        dictionary containing chain and resids of annotated region
    resid: int
        resid of residue
    chain: str
        chain of residue

    Returns
    -------
    bool
    """
    condition0 = (region.get('chain') is None or region.get('chain') == chain)
    condition1 = _in_resid_region(resid, region.get('resids'))

    return condition0 and condition1

