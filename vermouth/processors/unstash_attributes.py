"""
Provides a processor that unstashes attributes 
"""
import networkx as nx
from vermouth.processors.processor import Processor

def unstash_attributes(molecule, attributes, stash_name="stash"):
    """Unstash attributes from the stash.

    Parameters
    ----------
    molecule : vermouth.molecule.Molecule
        The molecule to unstash attributes for.
    attributes : Iterable[str]
        The attributes to unstash.
    stash_name : str
        Name of the node attribute where values were previously stashed.
        
    """
    for attribute in attributes:
        values = {node: data[stash_name][attribute]
                              for node, data in molecule.nodes(data=True)
                              if stash_name in data and attribute in data[stash_name]}
        nx.set_node_attributes(molecule, values, attribute)
        
class UnstashAttributes(Processor):
    def __init__(self, attributes=(), stash_name="stash"):
        self.attributes = attributes
        self.stash_name = stash_name
    def run_molecule(self, molecule):
        unstash_attributes(molecule, 
                           self.attributes, 
                           self.stash_name)
        return molecule
    