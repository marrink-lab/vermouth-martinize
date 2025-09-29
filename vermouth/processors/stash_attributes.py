#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 University of Groningen
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
Provides a processor that stores attributes 
"""

from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger
import copy
LOGGER = StyleAdapter(get_logger(__name__))


def stash_attributes(molecule, attributes, stash_name="stash"):
    """
    For each node in molecule, add the attributes to a stash dictionary

    Parameters
    ----------
    molecule: :class:`~vermouth.molecule.Molecule`
        The molecule to transform.
    attributes: tuple[str]
        Attributes to store in the nodes that may otherwise be modified
    stash_name: str
        Name of top level node dictionary to store attributes to
    """
    for node in molecule.nodes:
        for attr in attributes:
            # create the stash if not already there
            stash = molecule.nodes[node].get(stash_name, {})
            molecule.nodes[node][stash_name] = stash
            # stash the attribute if it hasn't already been. Otherwise raise warning.
            if attr not in stash:
                stash[attr] = copy.deepcopy(molecule.nodes[node].get(attr, None))
            else:
                LOGGER.warning("Trying to stash already stashed attribute {} to molecule. Will not stash.", attr)


class StashAttributes(Processor):
    """
    Processor for storing current attributes of a node in a new "stash" attribute

    attributes: tuple[str]
        Attributes to be stashed for later use.
    stash_name: str
        Name of top level node dictionary in which to store existing attributes
    """
    def __init__(self, attributes=(), stash_name='stash'):
        self.attributes = attributes
        self.stash_name = stash_name

    def run_molecule(self, molecule):
        stash_attributes(
            molecule,
            self.attributes,
            self.stash_name
        )
        return molecule

    def run_system(self, system):
        super().run_system(system)
