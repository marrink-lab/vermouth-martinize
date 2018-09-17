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
    template_from: dict
        Node template to select the atoms at one end of the edges.
    template_to: dict
        Node template to select the atoms at the other end of the edges.
    attribute: str
        Name of the attribute under which are stores the coordinates.

    See Also
    --------
    vermouth.molecule.attributes_match
    """
    def __init__(self, threshold, template_from, template_to, attribute='position'):
        self.threshold = threshold
        self.template_from = template_from
        self.template_to = template_to
        self.attribute = attribute

    def run_system(self, system):
        """
        Run the processor on the system.
        """
        system.molecules = edge_tuning.add_edges_threshold(
            system.molecules,
            threshold=self.threshold,
            template_a=self.template_from,
            template_b=self.template_to,
            attribute=self.attribute,
        )
        return system
