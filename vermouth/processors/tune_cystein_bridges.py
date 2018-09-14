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
Provides processors that can add and remove cystein bridges.
"""

import functools


from ..molecule import attributes_match
from .processor import Processor
from .. import edge_tuning

UNIVERSAL_BRIDGE_TEMPLATE = {'resname': 'CYS', 'atomname': 'SG'}


def remove_cystein_bridge_edges(molecule, template=UNIVERSAL_BRIDGE_TEMPLATE):  # pylint: disable=dangerous-default-value
    """
    Remove all the edges that correspond to cystein bridges from a molecule.

    Cystein bridge edges link an atom from a cystein side chain to the same
    atom on an other cystein. Selecting the correct atom is done with a
    template node dictionary, in the same way as node matching in links. The
    default template selects the 'SG' bead of the residue 'CYS':
    ``{'resname': 'CYS', 'atomname': 'SG'}``.

    A template is a dictionary that defines the key:value pairs that must be
    matched in the atoms. Values can be instances of
    :class:`~vermouth.molecule.LinkPredicate`.

    Parameters
    ----------
    molecule: networkx.Graph
        Molecule to modify in-place.
    template: dict
        A template that selected atom must match.
    """
    selector = functools.partial(attributes_match, template_attributes=template)
    edge_tuning.prune_edges_with_selectors(molecule, selector)


class RemoveCysteinBridgeEdges(Processor):
    def __init__(self, template=UNIVERSAL_BRIDGE_TEMPLATE):  # pylint: disable=dangerous-default-value
        self.template = template

    def run_molecule(self, molecule):
        remove_cystein_bridge_edges(molecule, self.template)
        return molecule


class AddCysteinBridgesThreshold(Processor):
    def __init__(self, threshold,  # pylint: disable=dangerous-default-value
                 template=UNIVERSAL_BRIDGE_TEMPLATE, attribute='position'):
        self.threshold = threshold
        self.template = template
        self.attribute = attribute

    def run_system(self, system):
        system.molecules = edge_tuning.add_edges_threshold(
            system.molecules, self.threshold,
            template_a=self.template,
            template_b=self.template,
            attribute=self.attribute
        )
        return system
