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
Provides a processor that renames residues based on their current residue names
and identified modifications, such as PTMs.
"""

from .processor import Processor


def rename_modified_residues(mol):
    """
    Renames residue names based on the current residue name, and the found
    modifications. The new names are found in
    `force_field.renamed_residues`, which should be a mapping of
    ``{(rename, [modification_name, ...]): new_name}``.

    Parameters
    ----------
    mol : Molecule
        The molecule whose residue names should be changed. Is modified
        in-place.
    """
    rename_map_ff = mol.force_field.renamed_residues
    rename_map = {}
    # Sort the list of modifications, so that the order does not matter. Don't
    # make a frozenset, because it might be possible to have the same
    # modification multiple times?
    # This should probably be done as the list is parsed. Also, as we parse it,
    # make sure we actually recognize all the modification names.
    for (resname, mods), new_name in rename_map_ff.items():
        rename_map[resname, tuple(sorted(mods))] = new_name

    for node_key in mol:
        node = mol.nodes[node_key]
        modifications = node.get('modifications', [])
        resname = node.get('resname', '')
        if not modifications:
            continue
        modifications = tuple(sorted(mod.graph['name'] for mod in modifications))
        try:
            new_name = rename_map[(resname, modifications)]
        except KeyError:
            # We don't know how to rename this residue, so continue to the next
            # node.
            continue
        node['resname'] = new_name


class RenameModifiedResidues(Processor):
    def run_molecule(self, molecule):
        rename_modified_residues(molecule)
        return molecule
