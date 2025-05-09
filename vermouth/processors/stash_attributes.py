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
Provides a processor that stores attributes 
"""

from .processor import Processor

def stash_attributes(molecule, attributes):
    """
    For each node in molecule, add the attributes to a stash dictionary
    """
    for attr in attributes:
        for node in molecule.nodes:
            # create the stash if not already there
            if not molecule.nodes[node].get("stash"):
                molecule.nodes[node]["stash"] = {}
            molecule.nodes[node]["stash"][attr] = molecule.nodes[node].get(attr, None)


class StashAttributes(Processor):
    """
    Processor for storing current attributes of a node in a new "stash" attribute

    """
    def __init__(self, attributes = ()):
        self.attributes = attributes

    def run_molecule(self, molecule):
        stash_attributes(
            molecule,
            self.attributes
        )
        return molecule

    def run_system(self, system):
        super().run_system(system)
