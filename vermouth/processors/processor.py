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

import networkx as nx

class Processor:
    """
    An abstract base class for processors. Subclasses must implement a
    `run_molecule` method.
    """
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

class Pipeline(nx.DiGraph, Processor):
    """
    A processor that executes an ordered collection of processors.

    Pipelines can contain both processors and nested pipelines.
    """

    def __init__(self, /, name=''):
        super().__init__()
        self.name = name

    @classmethod
    def from_dict(cls, conf, name):
        """
    Construct a Pipeline from a configuration dictionary.

    Parameters
    ----------
    conf: dict
        The pipeline configuration dictionary.
    name: str
        The name of the pipeline.

    Returns
    -------
    Pipeline
        The constructed pipeline.
    """
        
        def _recurse(parent, name, conf):
            if 'steps' in conf:
                obj = cls(name=name)
                for step_name, step in conf['steps']:
                    _recurse(obj, step_name, step)

            else:
                processor = conf.get('processor')

                if processor is None:
                    raise KeyError(f"Step {name} has no processor. Conf: {conf}")

                if not conf.get("condition", True):
                    obj = processor
                else:
                    kwargs = conf.get('args', {})
                    obj = processor(**kwargs)

            if parent is not None:
                parent.add(obj, condition=conf.get("condition", True))
            else:
                return obj

        return _recurse(None, name, conf)


    @property
    def processors(self):
        for idx in self.ordered_nodes:
            yield self.nodes[idx]['processor']

    @property
    def ordered_nodes(self):
        yield from nx.topological_sort(self)

    def add(self, processor, **kwargs):
        current = list(self.nodes)
        self.add_node(len(current), processor=processor, **kwargs)
        for idx in range(len(current)):
            self.add_edge(idx, len(current))

    def run_system(self, system):
        for node_idx in self.ordered_nodes:
            processor = self.nodes[node_idx]['processor']
            if isinstance(processor, type):
                name = processor.__name__
            else:
                name = getattr(processor, 'name', None) or processor.__class__.__name__
            if self.nodes[node_idx]['condition']:
                print(f'Running {name}')
                result = processor.run_system(system)
                if result is not None:
                    system = result
            else:
                print(f'Not running {name} because the condition is not met')
        return system

    def __str__(self):
        return "{name}[{members}]".format(name=self.name, members=', '.join(map(str, self.processors)))

    def __repr__(self):
        return str(self)
   
