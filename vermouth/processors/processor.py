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
Provides an abstract base class for processors.
"""
from ..log_helpers import StyleAdapter, get_logger

import networkx as nx

LOGGER = StyleAdapter(get_logger(__name__))

class Processor:
    """
    An abstract base class for processors. Subclasses must implement a
    `run_molecule` method.
    """
    def __str__(self):
        return self.__class__.__name__

    def run_system(self, system):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        mols = []
        for molecule in system.molecules:
            mols.append(self.run_molecule(molecule))
        system.molecules = mols

    def run_molecule(self, molecule):
        """
        Process a single molecule. Must be implemented by subclasses.

        Parameters
        ----------
        molecule: vermouth.molecule.Molecule
            The molecule to process.

        Returns
        -------
        vermouth.molecule.Molecule
            Either the provided molecule, or a brand new one.
        """
        raise NotImplementedError


class ProcessorPipeline(nx.DiGraph, Processor):
    def __init__(self, /, name=''):
        super().__init__()
        self.name = name or self.__class__.__name__

    @property
    def processors(self):
        order = nx.topological_sort(self)
        for idx in order:
            yield self.nodes[idx]['processor']

    def add(self, processor):
        current = list(self.nodes)
        self.add_node(len(current), processor=processor)
        for idx in range(len(current)):
            self.add_edge(idx, len(current))

    def run_system(self, system):
        for processor in self.processors:
            name = getattr(processor, 'name', None) or processor.__class__.__name__
            LOGGER.info(f"Running {name}", type='step')
            processor.run_system(system)

    def __str__(self):
        return "{name}[{members}]".format(name=self.name, members=', '.join(map(str, self.processors)))
