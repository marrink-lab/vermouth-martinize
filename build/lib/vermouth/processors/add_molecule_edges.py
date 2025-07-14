#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
Processor adding edges between molecules.
"""

from .processor import Processor
from .. import edge_tuning
from ..molecule import Choice

# TODO: this should be defined in the force fields
DNA_DONORS = (
    {'resname': Choice(['DA', 'DA3', 'DA5']), 'atomname': Choice(['C2', 'N6'])},
    {'resname': Choice(['DG', 'DG3', 'DG5']), 'atomname': Choice(['N1', 'N2'])},
    {'resname': Choice(['DC', 'DC3', 'DC5']), 'atomname': 'N4'},
    {'resname': Choice(['DT', 'DT3', 'DT5']), 'atomname': 'N3'},
)
DNA_ACCEPTORS = (
    {'resname': Choice(['DA', 'DA3', 'DA5']), 'atomname': 'N1'},
    {'resname': Choice(['DG', 'DG3', 'DG5']), 'atomname': 'O6'},
    {'resname': Choice(['DC', 'DC3', 'DC5']), 'atomname': Choice(['N3', 'O2'])},
    {'resname': Choice(['DT', 'DT3', 'DT5']), 'atomname': Choice(['O2', 'O4'])},
)
DNA_HB_DIST = 0.30


class AddMoleculeEdgesAtDistance(Processor):
    """
    Processor that adds edges within and between molecules.

    The processor adds edges between atoms, within or between molecules, when
    the atoms are part of the selections provided for each end of the edges,
    and the atoms are closer than a given threshold.

    Parameters
    ----------
    threshold: float
        Distance threshold in nanometers under which to create an edge.
    templates_from: list[dict]
        List of node templates to select the atoms at one end of the edges.
    templates_to: list[dict]
        List of node template to select the atoms at the other end of the edges.
    attribute: str
        Name of the attribute under which are stores the coordinates.

    See Also
    --------
    vermouth.molecule.attributes_match
    """
    def __init__(self, threshold, templates_from, templates_to,
                 attribute='position', min_edges=0):
        self.threshold = threshold
        self.templates_from = templates_from
        self.templates_to = templates_to
        self.attribute = attribute
        self.min_edges = min_edges

    def run_system(self, system):
        """
        Run the processor on the system.
        """
        system.molecules = edge_tuning.add_edges_threshold(
            system.molecules,
            threshold=self.threshold,
            templates_a=self.templates_from,
            templates_b=self.templates_to,
            attribute=self.attribute,
            min_edges=self.min_edges,
        )
        return system


class MergeNucleicStrands(AddMoleculeEdgesAtDistance):
    """
    Add edges between complementary nucleic acid strands.

    By default, the edges are added in place of the hydrogen bonds between
    complementary bases.

    Parameters
    ----------
    threshold: float
        Distance threshold in nanometers under which to create an edge.
    templates_donnors: list[dict]
        List of templates describing hydrogen donnors.
    templates_acceptors: list[dict]
        List of templates describing hydrogen acceptors.
    attribute: str
        Name of the attribute under which are store the node coordinates.
    """
    def __init__(self, threshold=DNA_HB_DIST,
                 templates_donnors=DNA_DONORS,
                 templates_acceptors=DNA_ACCEPTORS,
                 attribute='position'):
        super().__init__(
            threshold=threshold,
            templates_from=templates_donnors,
            templates_to=templates_acceptors,
            attribute=attribute,
            min_edges=4
        )
