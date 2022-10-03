#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2022 University of Groningen
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
Provides a Processor that sorts nodes and assigns them new, consecutive atomids.
The intended use is to make sure that residues are consecutive.
"""

from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger
LOGGER = StyleAdapter(get_logger(__name__))


class SortNodes(Processor):
    def __init__(self, groupby=('chain', 'resid', 'insertion_code', 'atomid'),
                 target_attr='atomid'):
        self.groupby_attributes = groupby
        self.target_attr = target_attr
        if self.groupby_attributes[-1] != self.target_attr:
            LOGGER.warning("{} is not the last element of {}. Atomid sorting may"
                           " be unstable.", self.target_attr, self.groupby_attributes)

    def run_molecule(self, molecule):
        def key_function(node_key):
            node = molecule.nodes[node_key]
            return tuple(node.get(attr) for attr in self.groupby_attributes)
        node_order = sorted(molecule.nodes, key=key_function)
        for atomid, node_key in enumerate(node_order, start=1):
            molecule.nodes[node_key][self.target_attr] = atomid
        return molecule