#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 University of Groningen
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
Provides processors that can add and remove IDR specific bonds
"""

from itertools import chain
from ..dssp.dssp import SS_CG, sequence_from_residues
from ..selectors import is_protein
from .processor import Processor
from ..rcsu.go_utils import _in_chain_and_resid_region
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


def parse_residues(resspec):
    """
    Parse a residue specification: [<chain>-][<resid_start>]:[<resid_end>] where
    resid is /[0-9]+/.
    Returns a dictionary with keys 'chain', 'resid_start', and 'resid_end' for the
    fields that are specified. Resids will be ints.

    Parameters
    ----------
    resspec: str

    Returns
    -------
    dict

    """
    # <chain>-<resid>
    *chain, resids = resspec.split('-', 1)
    res_start, res_end = resids.split(':', 1)

    out = {}
    if resids:
        out['resids'] = [(int(res_start), int(res_end))]
    if chain:
        out['chain'] = chain[0]
    else:
        out['chain'] = None
    return out

def annotate_disorder(molecule, id_regions, annotation="cgidr"):
    """
    Annotate the disordered regions of the molecule

    molecule: :class:`vermouth.molecule.Molecule`
            the molecule
    idr_regions: list
        dictionaries defining the disordered regions to annotate
    annotation: str
        name of the annotation in the node
    """
    for region in id_regions:
        for key, node in molecule.nodes.items():
            _old_resid = node['_old_resid']
            chain = node['chain']
            # make sure we have the correct chain and are in the right region. If no chain in region assume single chain system.
            if _in_chain_and_resid_region(region, _old_resid, chain):
                molecule.nodes[key][annotation] = True
                if "cgsecstruct" in molecule.nodes[key] and molecule.nodes[key]["cgsecstruct"] != "C":
                        molecule.nodes[key]["cgsecstruct"] = "C"
                        molecule.meta['modified_cgsecstruct'] = True
            else:
                molecule.nodes[key][annotation] = False



class AnnotateIDRs(Processor):
    """
    Processor to annotate intrinsically disordered regions of a molecule.

    This processor is designed primarily for the work described in the reference
    M3_GO, but is generally applicable for such circumstances where extra
    addition/removals are necessary.

    """

    def __init__(self, id_regions=None):
        """
        Parameters
        ----------
        id_regions:
            regions defining the IDRs
        """
        self.id_regions = []
        for region in id_regions:
            self.id_regions.append(parse_residues(region))

    def run_molecule(self, molecule):
        """
        Assign disordered regions for a single molecule
        """
        annotate_disorder(molecule, self.id_regions)
        return molecule

    def run_system(self, system):
        """
        Assign the water bias of the Go model to file. Biasing
        is always molecule specific i.e. no two different
        vermouth molecules can have the same bias.

        Parameters
        ----------
        system: :class:`vermouth.system.System`
        """
        if not self.id_regions:
            return system
        LOGGER.info("Annotating disordered regions.", type="step")
        super().run_system(system)

        if any([molecule.meta.get('modified_cgsecstruct', False) for molecule in system.molecules]):
            supplementary_ss_seq = list(
                chain(
                    *(
                        sequence_from_residues(molecule, "cgsecstruct")
                        for molecule in system.molecules
                        if is_protein(molecule)
                    )
                )
            )

            LOGGER.info(("Secondary structure assignment changed between dssp and martinize. "
                         "Check files for details."), type="general")
            system.meta["header"].extend((
                "The assigned secondary structure conflicted with ",
                "annotated IDRs. The following sequence of Martini secondary ",
                "structure was actually applied to the system:",
                "".join([SS_CG[i] for i in supplementary_ss_seq])
            ))
