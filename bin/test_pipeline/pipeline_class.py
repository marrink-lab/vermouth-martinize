from platform import system

import networkx as nx
import importlib
from vermouth.processors.processor import Processor


class Pipeline(nx.DiGraph, Processor):
    def __init__(self, /, name=''):
        super().__init__()
        self.name = name

    @classmethod
    def from_json_conf(cls, conf, name):
        def _recurse(parent, name, conf):
            if 'steps' in conf:
                obj = cls(name=name)
                for name, step in conf['steps']:
                    _recurse(obj, name, step)
            else:
                processor = conf['processor']
                kwargs = conf['args']
                obj = processor(**kwargs)
            if parent is not None:
                parent.add(obj, condition=conf['condition'])
            else:
                return obj

        self = _recurse(None, name, conf)
        return self


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
   