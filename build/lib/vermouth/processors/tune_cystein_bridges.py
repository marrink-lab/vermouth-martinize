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


from ..selectors import proto_multi_templates
from .processor import Processor
from .add_molecule_edges import AddMoleculeEdgesAtDistance
from .. import edge_tuning

UNIVERSAL_BRIDGE_TEMPLATE = [{'resname': 'CYS', 'atomname': 'SG'}, ]


def remove_cystein_bridge_edges(molecule, templates=UNIVERSAL_BRIDGE_TEMPLATE):  # pylint: disable=dangerous-default-value
    """
    Remove all the edges that correspond to cystein bridges from a molecule.

    Cystein bridge edges link an atom from a cystein side chain to the same
    atom on an other cystein. Selecting the correct atom is done with a list
    of template node dictionaries. A template node dictionary functions in the
    same way as node matching in links. An atom that can be involved in a
    cystein bridge must match at least one of the templates of the list.
    The default template list selects the 'SG' bead of the residue 'CYS':
    ``[{'resname': 'CYS', 'atomname': 'SG'}, ]``.

    A template is a dictionary that defines the key:value pairs that must be
    matched in the atoms. Values can be instances of
    :class:`~vermouth.molecule.LinkPredicate`.

    Parameters
    ----------
    molecule: networkx.Graph
        Molecule to modify in-place.
    templates: list[dict]
        A list of templates; selected atom must match at least one.
    """
    selector = functools.partial(proto_multi_templates, templates=templates)
    edge_tuning.prune_edges_with_selectors(molecule, selector)


class RemoveCysteinBridgeEdges(Processor):
    """
    Processor removing edges corresponding to cystein bridges.

    The edge for a cystein bridge is an edge between two atoms that match at
    least one template from a list of templates.

    Parameters
    ----------
    template: list[dict]
        List of node templates.
    """
    def __init__(self, template=UNIVERSAL_BRIDGE_TEMPLATE):  # pylint: disable=dangerous-default-value
        self.template = template

    def run_molecule(self, molecule):
        remove_cystein_bridge_edges(molecule, self.template)
        return molecule


class AddCysteinBridgesThreshold(AddMoleculeEdgesAtDistance):
    """
    Add edges corresponding to cystein bridges on a distance criterion.


    The edge for a cystein bridge is an edge between two atoms that match at
    least one template from a list of templates if the two ends of the edge are
    closer than a given distance.

    Parameters
    ----------
    threshold: float
        Distance in nanometers under which to consider an edge.
    template: list[dict]
        List of node templates.
    """
    def __init__(self, threshold,  # pylint: disable=dangerous-default-value
                 template=UNIVERSAL_BRIDGE_TEMPLATE, attribute='position'):
        super().__init__(
            threshold,
            templates_from=template,
            templates_to=template,
            attribute=attribute,
        )
