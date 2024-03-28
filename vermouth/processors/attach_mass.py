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
Provides a processor that assigns a `mass` attribute to every node in a
molecule based on it's element.
"""


from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))

# TODO: make the masses part of the forcefield
ATOM_MASSES = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 32, 'P': 31}
DEFAULT_MASS = 30

def attach_mass(molecule, attribute='mass'):
    """
    For every atom in `molecule` look up it's element in ``ATOM_MASSES``, and
    assign that value to `attribute`.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule to process. Is modified in-place.
    attribute: collections.abc.Hashable
        The attribute the mass is assigned to.
    """
    for node in molecule.nodes.values():
        element = node['element']
        if element not in ATOM_MASSES:
            LOGGER.info("Cannot find a mass for element {}. We assume it's some"
                        " metal, and will set its mass to {} amu.", element, DEFAULT_MASS)
        node[attribute] = ATOM_MASSES.get(node['element'], DEFAULT_MASS)


class AttachMass(Processor):
    def __init__(self, attribute='mass'):
        super().__init__()
        self.attribute = attribute

    def run_molecule(self, molecule):
        attach_mass(molecule, attribute=self.attribute)
        return molecule
