import numpy as np
from .processor import Processor


def do_average_bead(molecule):
    """
    Set the position of the particles to the mean of the underlying atoms.

    Parameters
    ----------
    molecule: martinize2.Molecule
        The molecule to update. The attribute :attr:`position` of the particles
        is updated on place. The nodes of the molecule must have an attribute
        :attr:`graph` that contains the subgraph of the initial molecule.
    """
    # Make sure the molecule fullfill the requirements.
    missing = []
    for node in molecule.nodes.values():
        if 'graph' not in node:
            missing.append(node)
    if missing:
        raise ValueError('{} particles are missing the graph attribute'
                         .format(len(missing)))

    for node in molecule.nodes.values():
        print([
            subnode
            for subnode in node['graph'].nodes().values()
        ])
        positions = np.stack([
            subnode['position']
            for subnode in node['graph'].nodes().values()
            if 'position' in subnode
        ])
        node['position'] = positions.mean(axis=0)

    return molecule


class DoAverageBead(Processor):
    def run_molecule(self, molecule):
        return do_average_bead(molecule)
