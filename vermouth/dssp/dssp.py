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
Assign protein secondary structures using DSSP.
"""

import collections
from functools import partial
import logging
import os
import subprocess
import tempfile
import re
import itertools
import mdtraj

from ..file_writer import deferred_open
from ..pdb import pdb
from ..system import System
from ..processors.processor import Processor
from ..processors import SortMoleculeAtoms
from ..selectors import is_protein, selector_has_position, filter_minimal, select_all
from .. import utils
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))

class DSSPError(Exception):
    """
    Exception raised if DSSP fails.
    """


def run_mdtraj(system):
    """
    Compute DSSP secondary structure assignments for the system by using
    ``mdtraj.compute_dssp``.

    During processing, a PDB file is produced. Therefore, all the molecules
    in the system must contain the required attributes for such a file to be
    generated. Also, the atom names are assumed to be compatible with the
    'charmm' force field for MDTraj to recognize them.
    However, the molecules do not require the edges to be defined.

    Parameters
    ----------
    system: System
        The system to process

    Returns
    -------
    list[str]
        The assigned secondary structures as a list of one-letter codes.
        The secondary structure sequences of all the molecules are combined
        in a single list without delimitation.
    """
    sys_copy = system.copy()
    # precaution for large systems; mdtraj requires all residues to be
    # grouped together otherwise dssp fails
    SortMoleculeAtoms(target_attr='atomid').run_system(sys_copy)
    tmpfile_handle, tmpfile_name = tempfile.mkstemp(suffix='.pdb', text=True,
                                                    dir='.', prefix='dssp_in_')
    tmpfile_handle = os.fdopen(tmpfile_handle, mode='w')
    tmpfile_handle.write(pdb.write_pdb_string(sys_copy, conect=False))
    tmpfile_handle.close()

    try:
        struct = mdtraj.load_pdb(tmpfile_name)
        dssp = mdtraj.compute_dssp(struct, simplified=False)
    except Exception as error:
        # Don't delete the temporary file
        message = "MDTraj encountered an error. The message was {err}. "\
                  "The input file provided to MDTraj can be found at {file}."
        raise DSSPError(message.format(err=str(error), file=tmpfile_name)) from error
    else:
        dssp = ['C' if ss == ' ' else ss for mol in dssp for ss in mol]
        if LOGGER.getEffectiveLevel() > logging.DEBUG:
            os.remove(tmpfile_name)
    return dssp


def annotate_dssp(molecule, callable=None, attribute='secstruct'):
    """
    Adds the DSSP assignation to the atoms of a molecule.

    Runs DSSP on the molecule and adds the secondary structure assignation as
    an attribute of its atoms. The attribute name in which the assignation is
    stored is controlled with the "attribute" argument.

    Only proteins can be annotated. Non-protein molecules are returned
    unmodified, so are empty molecules, and molecules for which no positions
    are set.

    The atom names are assumed to be compatible with DSSP. Atoms with no known
    position are not passed to DSSP which may lead to an error in DSSP.

    .. warning::

        The molecule is annotated **in-place**.

    Parameters
    ----------
    molecule: Molecule
        The molecule to annotate. Its atoms must have the attributes required
        to write a PDB file; other atom attributes, edges, or molecule
        attributes are not used.
    callable: Callable
        The function to call to generate DSSP secondary structure assignments.
        See also:  :func:`run_mdtraj`
    attribute: str
        The name of the atom attribute in which to store the annotation.

    See Also
    --------
    run_mdtraj
    """
    if not is_protein(molecule):
        return

    clean_pos = molecule.subgraph(
        filter_minimal(molecule, selector=selector_has_position)
    )

    # We ignore empty molecule, there is no point at running DSSP on them.
    if not clean_pos:
        return

    system = System()
    system.add_molecule(clean_pos)
    secstructs = callable(system)

    annotate_residues_from_sequence(molecule, attribute, secstructs)


def convert_dssp_to_martini(sequence):
    """
    Convert a sequence of secondary structure to martini secondary sequence.

    Martini treats some secondary structures with less resolution than dssp.
    For instance, the different types of helices that dssp discriminates are
    seen the same by martini. Yet, different parts of the same helix are seen
    differently in martini.

    In the Martini force field, the B and E secondary structures from DSSP are
    both treated as extended regions. All the DSSP helices are treated the
    same, but the different part of the helices (beginning, end, core of a
    short helix, core of a long helix) are treated differently.

    After the conversion, the secondary structures are:
    * :F: Collagenous Fiber
    * :E: Extended structure (β sheet)
    * :H: Helix structure
    * :1: Helix start (H-bond donor)
    * :2: Helix end (H-bond acceptor)
    * :3: Ambivalent helix type (short helices)
    * :T: Turn
    * :S: Bend
    * :C: Coil

    Parameters
    ----------
    sequence: str
        A sequence of secondary structures as read from dssp. One letter per
        residue.

    Returns
    -------
    str
        A sequence of secondary structures usable for martini. One letter per
        residue.
    """
    ss_cg = {'1': 'H', '2': 'H', '3': 'H', 'H': 'H', 'G': 'H', 'I': 'H',
             'B': 'E', 'E': 'E', 'T': 'T', 'S': 'S', 'C': 'C'}
    patterns = collections.OrderedDict([
        ('.H.', '.3.'), ('.HH.', '.33.'), ('.HHH.', '.333.'),
        ('.HHHH.', '.3333.'), ('.HHHHH.', '.13332.'),
        ('.HHHHHH.', '.113322.'), ('.HHHHHHH.', '.1113222.'),
        ('.HHHH', '.1111'), ('HHHH.', '2222.'),
    ])
    cg_sequence = ''.join(ss_cg[secstruct] for secstruct in sequence)
    wildcard_sequence = ''.join('H' if secstruct == 'H' else '.'
                                for secstruct in cg_sequence)
    # Flank the sequence with dots. Otherwise in a sequence consisting of only
    # H will not have a start or end. See also issue 566.
    # This should not cause further issues, since '..' doesn't map to anything
    wildcard_sequence = '.' + wildcard_sequence + '.'
    for pattern, replacement in patterns.items():
        while pattern in wildcard_sequence:  # EXPENSIVE! :'(
            wildcard_sequence = wildcard_sequence.replace(pattern, replacement)
    # And remove the flanking dots again
    wildcard_sequence = wildcard_sequence[1:-1]
    result = ''.join(
        wildcard if wildcard != '.' else cg
        for wildcard, cg in zip(wildcard_sequence, cg_sequence)
    )
    return result


def sequence_from_residues(molecule, attribute, default=None):
    """
    Generates a sequence of `attribute`, one per residue in `molecule`.

    Parameters
    ----------
    molecule: vermouth.molecule.Molecule
        The molecule to process.
    attribute: collections.abc.Hashable
        The attribute of interest.
    default: object
        Yielded if the first node of a residue has no attribute `attribute`.

    Yields
    ------
    object
        The value of `attribute` for every residue in `molecule`.
    """
    for residue_nodes in molecule.iter_residues():
        # TODO: Make sure they're the same for every node in residue.
        first_name = residue_nodes[0]
        first_node = molecule.nodes[first_name]
        value = first_node.get(attribute, default)
        yield value


def annotate_residues_from_sequence(molecule, attribute, sequence):
    """
    Sets the attribute `attribute` to a value from `sequence` for every node in
    `molecule`. Nodes in the n'th residue of `molecule` are given the n'th
    value of `sequence`.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule to annotate. Is modified in-place.
    attribute: collections.abc.Hashable
        The attribute to set.
    sequence: collections.abc.Sequence
        The values assigned.

    Raises
    ------
    ValueError
        If the length of `sequence` is different from the number of residues in
        `molecule`.
    """
    residues = list(molecule.iter_residues())
    if len(sequence) == 1:
        sequence = sequence * len(residues)
    elif len(sequence) != len(residues):
        msg = ('The sequence length does not match the number of residues. '
               'The sequence has {} elements for {} residues.')
        raise ValueError(msg.format(len(sequence), len(residues)))
    for residue_nodes, value in zip(residues, sequence):
        for node_name in residue_nodes:
            molecule.nodes[node_name][attribute] = value


def convert_dssp_annotation_to_martini(
        molecule, from_attribute='secstruct', to_attribute='cgsecstruct'):
    """
    For every node in `molecule`, translate the `from_attribute` with
    :func:`convert_dssp_to_martini`, and assign it to the attribute
    `to_attribute`.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule to process. Is modified in-place.
    from_attribute: collections.abc.Hashable
        The attribute to read.
    to_attribute: collections.abc.Hashable
        The attribute to set.

    Raises
    ------
    ValueError
        If not all nodes have a `from_attribute`.
    """
    dssp_sequence = list(sequence_from_residues(molecule, from_attribute))
    if None not in dssp_sequence:
        cg_sequence = list(convert_dssp_to_martini(dssp_sequence))
        annotate_residues_from_sequence(molecule, to_attribute, cg_sequence)
    elif all(elem is None for elem in dssp_sequence):
        # There is no DSSP assignation for the molecule. This is likely due to
        # the molecule not being a protein. Anyway, we issue a debug message
        # as it *could* be due to the DSSP assignation having been skipped
        # for some reason.
        msg = 'No DSSP assignation to convert to Martini secondary structure intermediates.'
        LOGGER.debug(msg)
    else:
        # This is more of a problem. For now, we do not know what to do with
        # incomplete DSSP assignation. This may come later as a problem if
        # a molecule is attached to a protein.
        raise ValueError('Not all residues have a DSSP assignation.')


def gmx_system_header(system):

    ss_sequence = list(
        itertools.chain(
            *(
                sequence_from_residues(molecule, "secstruct")
                for molecule in system.molecules
                if is_protein(molecule)
            )
        )
    )

    if None not in ss_sequence and ss_sequence:
        system.meta["header"].extend(("The following sequence of secondary structure ",
                                      "was used for the full system:",
                                      "".join(ss_sequence),
                                     ))

class AnnotateDSSP(Processor):
    name = 'AnnotateDSSP'

    def __init__(self):
        super().__init__()

    def run_molecule(self, molecule):
        annotate_dssp(molecule, run_mdtraj)
        molecule.citations.add('MDTraj')
        return molecule


class AnnotateMartiniSecondaryStructures(Processor):
    name = 'AnnotateMartiniSecondaryStructures'

    @staticmethod
    def run_molecule(molecule):
        convert_dssp_annotation_to_martini(molecule)
        return molecule

    def run_system(self, system):
        gmx_system_header(system)
        super().run_system(system)


class AnnotateResidues(Processor):
    """
    Set an attribute of the nodes from a sequence with one element per residue.

    Read a sequence with one element per residue and assign an attribute of
    each node based on that sequence, so each node has the value corresponding
    to its residue. In most cases, the length of the sequence has to match the
    total number of residues in the system. The sequence must be ordered in the
    same way as the residues in the system. If all the molecules have the same
    number of residues, and if the length of the sequence corresponds to the
    number of residue of one molecule, then the sequence is repeated to all
    molecules. If the sequence contains only one element, then it is repeated
    to all the residues ofthe system.

    Parameters
    ----------
    attribute: str
        Name of the node attribute to populate.
    sequence: collections.abc.Sequence
        Per-residue sequence.
    molecule_selector: collections.abc.Callable
        Function that takes an instance of :class:`vermouth.molecule.Molecule`
        as argument and returns `True` if the molecule should be considered,
        else `False`.
    """
    name = 'AnnotateResidues'

    def __init__(self, attribute, sequence,
                 molecule_selector=select_all):
        self.attribute = attribute
        self.sequence = sequence
        self.molecule_selector = molecule_selector

    def run_molecule(self, molecule):
        """
        Run the processor on a single molecule.

        Parameters
        ----------
        molecule: vermouth.molecule.Molecule

        Returns
        -------
        vermouth.molecule.Molecule
        """
        if self.molecule_selector(molecule):
            annotate_residues_from_sequence(molecule, self.attribute, self.sequence)
        return molecule

    def run_system(self, system):
        """
        Run the processor on a system.

        Parameters
        ----------
        system: vermouth.system.System

        Returns
        -------
        vermouth.system.System
        """
        # Test and adjust the length of the sequence. There are 3 valid scenarios:
        # * the length of the sequence matches the number of residues in the
        #   selection;
        # * all the molecules in the selection have the same number of residues
        #   and the sequence length matches the number of residue of one
        #   molecule; in this case the equence is repeated for each molecule;
        # * the sequence has a length of one; in this case the sequence is
        #   repeated for each residue.
        # The case were there is no molecule in the selection is only valid if
        # the sequence is empty. Then we are in the first valid scenario.
        molecule_lengths = [
            len(list(molecule.iter_residues()))
            for molecule in system.molecules
            if self.molecule_selector(molecule)
        ]
        if self.sequence and not molecule_lengths:
            raise ValueError('There is no molecule to which '
                             'to apply the sequence.')
        if (molecule_lengths
                and len(self.sequence) == molecule_lengths[0]
                and utils.are_all_equal(molecule_lengths)):
            sequence = list(self.sequence) * len(molecule_lengths)
        elif len(self.sequence) == 1:
            sequence = list(self.sequence) * sum(molecule_lengths)
        elif len(self.sequence) != sum(molecule_lengths):
            raise ValueError(
                'The length of the sequence ({}) does not match the '
                'number of residues in the selection ({}).'
                .format(len(self.sequence), sum(molecule_lengths))
            )
        else:
            sequence = self.sequence

        end = 0
        begin = 0
        for molecule, nres in zip(system.molecules, molecule_lengths):
            end += nres
            annotate_residues_from_sequence(
                molecule,
                self.attribute,
                sequence[begin:end]
            )
            begin += nres
