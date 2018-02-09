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
import itertools
import os
import subprocess

from ..pdb import pdb
from ..system import System
from ..molecule import Molecule
from ..processors.processor import Processor


PROTEIN_RESIDUES = ('ALA', 'ARG', 'ASP', 'ASN', 'CYS',
                    'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL')


class DSSPError(Exception):
    """
    Exception raised if DSSP fails.
    """
    pass


def is_protein(molecule):
    """
    Return True is all the residues in the molecule are protein residues.

    The function tests if the residue name of all the atoms in the input
    molecule are in ``PROTEIN_RESIDUES``.

    Parameters
    ----------
    molecule: Molecule
        The molecule to test.

    Returns
    -------
    bool
    """
    for atom in molecule.nodes.values():
        resname = atom.get('resname')
        if resname not in PROTEIN_RESIDUES:
            return False
    return True


def selector_no_position(atom):
    """
    Return True if the atom does not have a position.

    An atom is considered as not having a position if:
    * the "position" key is not defined;
    * the value of "position" is ``None``.

    Parameters
    ----------
    atom: dict

    Returns
    -------
    bool
    """
    return atom.get('position') is None


def filter_out(molecule, selector):
    """
    Create a minimal molecule which **exclude** a selection.

    The minimal molecule only has nodes. It does dot have any molecule level
    attribute, not does it have edges.

    The selector must be a function that accepts an atom as a argument. The
    atom is passed as a node attribute dictionary. The selector must return
    ``True`` for atoms that are part of the selection to **exclude**.

    Parameters
    ----------
    molecule: Molecule
    selector: callback

    Returns
    -------
    Molecule
    """
    filtered = Molecule()
    for name, atom in molecule.nodes.items():
        if not selector(atom):
            filtered.add_node(name, **atom)
    return filtered


def read_dssp2(lines):
    """
    Read the secondary structure from a DSSP output.

    Only the first column of the "STRUCTURE" block is read. See the
    `documentation of the DSSP format`_ for more details.

    The secondary structures that can be read are:

    :H: α-helix
    :B: residue in isolated β-bridge
    :E: extended strand, participates in β ladder
    :G: 3-helix (3-10 helix)
    :I: 5 helix (π-helix)
    :T: hydrogen bonded turn
    :S: bend
    :C: loop or irregular

    The "C" code for loops and random coil is translated from the gap used in
    the DSSP file for an improved readability.

    Only the version 2 of DSSP is supported. If the format is not recognized as
    comming from that version of DSSP, then a :exc:`IOError` is raised.

    .. _`documentation of the DSSP format`: http://swift.cmbi.ru.nl/gv/dssp/DSSP_3.html

    Parameters
    ----------
    lines:
        An iterable over the lines of the DSSP output. This can be *e.g.* a
        list of lines, or a file handler. The new line character is ignored.

    Returns
    -------
    secstructs: list of str
        The secondary structure assigned by DSSP as a list of one-letter
        secondary structure code.

    Raises
    ------
    IOError
        When a line could not be parsed, or if the version of DSSP
        is not supported.
    """
    secstructs = []
    # We want to be able to read the first line in the same way regardless of
    # the type of the input. As we cannot read the first element of a list
    # using `next`, nor can we read the first line of a file by indexing,
    # we normalize the interface of the input by making it an iterator.
    # It should always be possible to make an iterator out of a iterable.
    lines = iter(lines)
    first_line = next(lines)
    # The function can only read output from DSSP version 2. Hopefully, if the
    # input file is not in this format, then the parser will break as it reads
    # the file; we can expect that the end of the header will not be found or
    # the secondary structure will be an unexpected character.
    # We could predict from the first line that the format is not the one we
    # expect if it does not start with "===="; however, the first lines of the
    # file are non-essential and could have been trimmed. (For instance, the
    # first line of a DSSPv2 file contains the date of execution of the
    # program, which is annoying when comparing files.) Failing at the
    # first line is likely unnecessary.
    # Yet, we can identify files from DSSP v1 from the first line. These files
    # start with "****" instead of "====". If we identify such a file, we can
    # fail with a useful error message.
    if first_line and not first_line.startswith('===='):
        if first_line.startswith('****'):
            msg = ('Based on its header, the input file could come from a '
                   'pre-July 1995 version of DSSP (or the compatibility mode '
                   'of a more recent version). Only output from the version 2 '
                   'of DSSP are supported.')
            raise IOError(msg)

    # First we skip the header and the histogram.
    line_num = 0  # This should not be required, except maybe weird edge-cases
    for line_num, line in enumerate(lines, start=2):
        if line[:15] == '  #  RESIDUE AA':
            break
    else:  # no break
        msg = ('No secondary structure assignation could be read because the '
               'file is not formated correctly. No line was found that starts '
               'with "  #  RESIDUE AA".')
        raise IOError(msg)

    # Now, every line should be a secondary structure assignation.
    for line_num, line in enumerate(lines, start=line_num + 1):
        if '!' in line:
            # This is a TER record, we ignore it.
            continue
        if not line:
            # We ignore the empty lines.
            continue
        if len(line) >= 17:
            secondary_structure = line[16]
            if secondary_structure not in 'HBEGITS ':
                msg = 'Unrecognize secondary structure "{}" in line {}: "{}"'
                raise IOError(msg.format(secondary_structure, line_num, line))
            # DSSP represents the coil with a space. While this works in a
            # column based file, it is much less convenient to handle in
            # our code, and it is much less readable in our debug logs.
            # We translate the space to "C" in our representation.
            if secondary_structure == ' ':
                secondary_structure = 'C'
            secstructs.append(secondary_structure)
        else:
            raise IOError('Line {} is too short: "{}".'.format(line_num, line))

    return secstructs


def run_dssp(system, executable='dssp', savefile=None):
    """
    Run DSSP on a system and return the assigned secondary structures.

    Run DSSP using the path (or name in the research PATH) given by
    "executable". Return the secondary structure parsed from the output of the
    program.

    In order to call DSSP, a PDB file is produced. Therefore, all the molecules
    in the system must contain the required attributes for such a file to be
    generated. Also, the atom names are assumed to be compatible with them
    'universal' force field for DSSP to recognize them.
    However, the molecules do not require the edges to be defined.

    DSSP is assumed to be in version 2. The secondary structure codes are
    described in :fun:`read_dssp2`.

    If "savefile" is set to a path, then the output of DSSP is written in
    that file.

    Parameters
    ----------
    system: System
    executable: str
        Where to find the DSSP executable.
    savefile: None or path
        If set to a path, the output of DSSP is written in that file.

    Returns
    list of str
        The assigned secondary structures as a list of one-letter codes.
        The secondary structure sequences of all the molecules are combined
        in a single list without delimitation.

    Raises
    ------
    DSSPError
        DSSP failed to run.
    IOError
        The output of DSSP could not be parsed.

    See Also
    --------
    read_dssp2
        Parse a DSSP output.
    """
    process = subprocess.Popen(
        [executable, "-i", "/dev/stdin"],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE
    )
    process.stdin.write(pdb.write_pdb_string(system, conect=False).encode('utf8'))
    out, err = process.communicate()
    out = out.decode('utf8')
    status = process.wait()
    if status:
        raise DSSPError(err.decode('utf8'))
    if savefile is not None:
        with open(savefile, 'w') as outfile:
            outfile.write(out)
    return read_dssp2(out.split('\n'))


def _savefile_path(molecule, savedir=None):
    savefile = None
    if savedir is not None:
        first_atom = list(molecule.nodes.keys())[0]
        chain = molecule.nodes[first_atom].get('chain')
        if chain is None:
            msg = 'The "savedir" argument can only be used if chains are set.'
            raise ValueError(msg)
        savefile = os.path.join(savedir, 'chain_{}.ssd'.format(chain))
    return savefile


def annotate_dssp(molecule, executable='dssp', savedir=None, attribute='secstruct'):
    """
    Adds the DSSP assignation to the atoms of a molecule.

    Runs DSSP on the molecule and adds the secondary structure assignation as
    an attribute of its atoms. The attribute name in which the assignation is
    stored is controlled with the "attribute" argument.

    Only proteins can be annotated. Non-protein molecules are returned
    unmodified, so as empty molecules, and molecules for which no positions
    are set.

    The atom names are assumed to be compatible with DSSP. Atoms with no known
    position are not passed to DSSP which may lead to an error in DSSP.

    .. warning::

        The molecule is annotated **on-place**.

    Parameters
    ----------
    molecule: Molecule
        The molecule to annotate. Its atoms must have the attributes required
        to write a PDB file; other atom attributes, edges, or molecule
        attributes are not used.
    executable: str
        The path or name in the research PATH of the DSSP executable.
    savedir: None or str
        If set to a path, the DSSP output will be written in this **directory**.
        The option is only available if chains are defined with the 'chain'
        atom attribute.
    attribute: str
        The name of the atom attribute in which to store the annotation.

    Returns
    -------
    Molecule
        The modified molecule.

    See Also
    --------
    run_dssp, read_dssp2
    """
    if not is_protein(molecule):
        return molecule

    clean_pos = filter_out(molecule, selector=selector_no_position)

    # We ignore empty molecule, there is no point at running DSSP on them.
    if not clean_pos:
        return molecule

    savefile = _savefile_path(molecule, savedir)

    system = System()
    system.add_molecule(clean_pos)
    secstructs = run_dssp(system, executable, savefile)

    group_residues = itertools.groupby(molecule, key=lambda atom: molecule.nodes[atom]['resid'])
    for (_, atoms), secondary_structure in zip(group_residues, secstructs):
        for atom in atoms:
            molecule.nodes[atom][attribute] = secondary_structure
    return molecule


def convert_dssp_to_martini(sequence):
    """
    Convert a sequence of secondary structure to martini secondary sequence.

    Martini treats some secondary structures with less resolution than dssp.
    For instance, the different types of helices that dssp discriminates are
    seen the same by martini. Yet, different parts of the same helix are seen
    differently in martini.

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
    for pattern, replacement in patterns.items():
        wildcard_sequence = wildcard_sequence.replace(pattern, replacement)
    result = ''.join(
        wildcard if wildcard != '.' else cg
        for wildcard, cg in zip(wildcard_sequence, cg_sequence)
    )
    return result


def sequence_from_residues(molecule, attribute, default=None):
    sequence = []
    for residue_nodes in molecule.iter_residues():
        first_name = residue_nodes[0]
        first_node = molecule.nodes[first_name]
        value = first_node.get(attribute, default)
        sequence.append(value)
    return sequence


def annotate_residues_from_sequence(molecule, attribute, sequence):
    for residue_nodes, value in zip(molecule.iter_residues(), sequence):
        for node_name in residue_nodes:
            node = molecule.nodes[node_name][attribute] = value


def convert_dssp_annotation_to_martini(
        molecule, from_attribute='secstruct', to_attribute='cgsecstruct'):
    dssp_sequence = sequence_from_residues(molecule, from_attribute)
    if None not in dssp_sequence:
        cg_sequence = list(convert_dssp_to_martini(dssp_sequence))
        annotate_residues_from_sequence(molecule, to_attribute, cg_sequence)


class AnnotateDSSP(Processor):
    name = 'AnnotateDSSP'

    def __init__(self, executable='dssp', savedir=None):
        super().__init__()
        self.executable = executable
        self.savedir = savedir

    def run_molecule(self, molecule):
        return annotate_dssp(molecule, self.executable, self.savedir)


class AnnotateMartiniSecondaryStructures(Processor):
    name = 'AnnotateMartiniSecondaryStructures'

    def run_molecule(self, molecule):
        convert_dssp_annotation_to_martini(molecule)
        return molecule
