import collections
import itertools
import networkx as nx

from ..molecule import Block, Link

__all__ = ['read_rtp']

# Name of the subsections in RTP files.
# Names starting with a '_' are for internal use.
RTP_SUBSECTIONS = ('atoms', 'bonds', 'angles', 'dihedrals',
                   'impropers', 'cmap', 'exclusions',
                   '_bondedtypes')


_BondedTypes = collections.namedtuple(
    '_BondedTypes', 
    'bonds angles dihedrals impropers all_dihedrals nrexcl HH14 remove_dih'
)


class _IterRTPSubsectionLines(object):
    """
    Iterate over the lines of an RTP file within a subsection.
    """
    def __init__(self, parent):
        self.parent = parent
        self.lines = parent.lines
        self.running = True

    def __next__(self):
        if not self.running:
            raise StopIteration
        line = next(self.lines)
        if line.strip()[0] == '[':
            self.parent.buffer.append(line)
            self.running = False
            raise StopIteration
        return line

    def __iter__(self):
        return self

    def flush(self):
        for line in self:
            pass


class _IterRTPSubsections(object):
    """
    Iterate over the subsection of a RTP file within a section.

    For each subsections, yields its name and  an iterator over its lines.
    """
    def __init__(self, parent):
        self.parent = parent
        self.lines = parent.lines
        self.buffer = collections.deque()
        self.current_subsection = None
        self.running = True

    def __next__(self):
        if not self.running:
            raise StopIteration
        if self.current_subsection is not None:
            self.current_subsection.flush()
        if self.buffer:
            line = self.buffer.popleft()
        else:
            line = next(self.lines)
        stripped = line.strip()
        if stripped[0] == '[':
            name = stripped[1:-1].strip()
            if name in RTP_SUBSECTIONS:
                subsection = _IterRTPSubsectionLines(self)
                self.current_subsection = subsection
                return name, subsection
            self.parent.buffer.append(line)
            self.running = False
            raise StopIteration
        print(self, line)
        raise IOError('I am almost sure I should not be here...')

    def __iter__(self):
        return self

    def flush(self):
        for line in self:
            pass


class _IterRTPSections(object):
    """
    Iterate over the sections of a RTP file.

    For each section, yields the name of the sections and an iterator over its
    subsections.
    """
    def __init__(self, lines):
        self.lines = lines
        self.buffer = collections.deque()
        self.current_section = None

    def __next__(self):
        if self.current_section is not None:
            self.current_section.flush()
        if self.buffer:
            line = self.buffer.popleft()
        else:
            line = next(self.lines)
        stripped = line.strip()
        if stripped[0] == '[':
            name = stripped[1:-1].strip()
            section = _IterRTPSubsections(self)
            self.current_section = section
            # The "[ bondedtypes ]" is special in the sense that it does
            # not have subsection, but instead have its content directly
            # under it. This breaks the neat hierarchy the rest of the file
            # has. Here we restore the hierarchy by faking that the file
            # contains:
            #
            #    [ bondedtypes ]
            #     [ _bondedtypes ]
            #
            # For now, I do that on the basis of the section name. If other
            # sections are special that way, I'll detect them with a look
            # ahead.
            if name == 'bondedtypes':
                section.buffer.append(' [ _bondedtypes ]')
            return name, section
        print(self, line)
        raise IOError('Hum... There is a bug in the RTP reader.')

    def __iter__(self):
        return self



def _atoms(subsection, block):
    for line in subsection:
        name, atype, charge, charge_group = line.split()
        atom = {
            'name': name,
            'atype': atype,
            'charge': float(charge),
            'charge_group': int(charge_group),
        }
        block.add_atom(atom)


def _base_rtp_parser(interaction_name, natoms):
    def wrapped(subsection, block):
        interactions = []
        for line in subsection:
            splitted = line.strip().split()
            atoms = splitted[:natoms]
            parameters = splitted[natoms:]
            interactions.append({'atoms': atoms, 'parameters': parameters})
        block.interactions[interaction_name] = interactions
    return wrapped

def _parse_bondedtypes(section):
    # Default taken from
    # 'src/gromacs/gmxpreprocess/resall.cpp::read_resall' in the Gromacs
    # source code.
    defaults = _BondedTypes(bonds=1, angles=1, dihedrals=1,
                            impropers=1, all_dihedrals=0,
                            nrexcl=3, HH14=1, remove_dih=1)

    # The 'bondedtypes' section contains its line directly under it. In
    # order to match the hierarchy model of the rest of the file, the
    # iterator actually yields a subsection named '_bondedtypes'. We need
    # to read the fist line of that first virtual subsection.
    _, lines = next(section)
    line = next(lines)
    read = [int(x) for x in line.split()]

    # Fill with the defaults. The file gives the values in order so we
    # need to append the missing values from the default at the end.
    bondedtypes = _BondedTypes(*(read + list(defaults[len(read):])))
    
    # Make sure there is no unexpected lines in the section.
    # Come on Jonathan! There must be a more compact way of doing it.
    try:
        next(lines)
    except StopIteration:
        pass
    else:
        raise IOError('"[ bondedtypes ]" section is missformated.')
    try:
        next(section)
    except StopIteration:
        pass
    else:
        raise IOError('"[ bondedtypes ]" section is missformated.')
    return bondedtypes


def _count_hydrogens(names):
    return len([name for name in names if name[0] == 'H'])


def _keep_dihedral(center, block, bondedtypes):
    if (not bondedtypes.all_dihedrals) and block.has_dihedral_around(center):
        return False
    if bondedtypes.remove_dih and block.has_improper_around(center):
        return False
    return True


def _complete_block(block, bondedtypes):
    """
    Add information from the bondedtypes section to a block.

    Generate implicit dihedral angles, and add function types to the
    interactions.
    """
    block._make_edges()

    # Generate missing dihedrals
    # As pdb2gmx generates all the possible dihedral angles by default,
    # RTP files are written assuming they will be generated. A RTP file
    # have some control over these dihedral angles through the bondedtypes
    # section.
    all_dihedrals = []
    for center, dihedrals in itertools.groupby(
            sorted(block.guess_dihedrals(), key=_dihedral_sorted_center),
            _dihedral_sorted_center):
        if _keep_dihedral(center, block, bondedtypes):
            # TODO: Also sort the dihedrals by index.
            # See src/gromacs/gmxpreprocess/gen_add.cpp::dcomp in the
            # Gromacs source code (see version 2016.3 for instance).
            atoms = sorted(dihedrals, key=_count_hydrogens)[0]
            all_dihedrals.append({'atoms': atoms})
    # TODO: Sort the dihedrals by index
    block.interactions['dihedrals'] = (
        block.interactions.get('dihedrals', []) + all_dihedrals
    )

    # TODO: generate 1-4 interactions between pairs of hydrogen atoms

    # Add function types to the interaction parameters. This is done as a
    # post processing step to cluster as much interaction specific code
    # into this method.
    # I am not sure the function type can be set explicitly in the RTP
    # file except through the bondedtypes section. This way of handling 
    # the function types would result in the function type being written
    # twice in that case. Yet, none of the RTP files distributed with
    # Gromacs 2016.3 case issue.
    functypes = {
        'bonds': bondedtypes.bonds,
        'angles': bondedtypes.angles,
        'dihedrals': bondedtypes.dihedrals,
        'impropers': bondedtypes.impropers,
        'exclusions': 1,
        'cmap': 1,
    }
    for name, interactions in block.interactions.items():
        for interaction in interactions:
            if 'parameters' in interaction:
                interaction['parameters'].insert(0, functypes[name])
            else:
                interaction['parameters'] = [functypes[name], ]


def _split_blocks_and_links(pre_blocks):
    """
    Split all the pre-blocks from `pre_block` into blocks and links.

    Parameters
    ----------
    pre_blocks: dict
        A dict with residue names as keys and instances of :class:`Block`
        as values.

    Returns
    -------
    blocks: dict
        A dict like `pre_block` with all inter-residues information
        stripped out.
    links: list
        A list of instances of :class:`Link` containing the inter-residues
        information from `pre_blocks`.

    See Also
    --------
    _split_block_and_link
        Split an individual pre-block into a block and a link.
    """
    blocks = {}
    links = []
    for name, pre_block in pre_blocks.items():
        block, link = _split_block_and_link(pre_block)
        blocks[name] = block
        links.append(link)
    return blocks, links


def _split_block_and_link(pre_block):
    """
    Split `pre_block` into a block and a link.

    A pre-block is a block as read from an RTP file. It contains both
    intra-residue and inter-residues information. This method split this
    information so that the intra-residue interactions are put in a block
    and the inter-residues ones are put in a link.

    Parameters
    ----------
    pre_block: Block
        The block to split.

    Returns
    -------
    block: Block
        All the intra-residue information.
    link: Link
        All the inter-residues information.
    """
    block = Block()
    link = Link()

    # It is easier to fill the interactions using a defaultdict,
    # yet defaultdicts are more annoying when reading as querying them
    # creates the keys. So the interactions are revert back to regular
    # dict at the end.
    block.interactions = collections.defaultdict(list)
    link.interactions = collections.defaultdict(list)

    block.name = pre_block.name
    
    # Filter the particles from neighboring residues out of the block.
    for atom in pre_block.atoms:
        if not atom['name'][0] in '+-':
            block.add_atom(atom)
    
    # Create the edges of the link based on the bonds in the pre-block.
    # This will create too many edges, but the useless ones will be pruned
    # latter.
    for bond in pre_block.interactions.get('bonds', []):
        link.add_edge(*bond['atoms'])

    # Split the interactions from the pre-block between the block (for
    # intra-residue interactions) and the link (for inter-residues ones).
    # The "relevant_atoms" set keeps track of what particles are
    # involved in the link. This will allow to prune the link without
    # iterating again through its interactions.
    relevant_atoms = set()
    for name, interactions in pre_block.interactions.items():
        for interaction in interactions:
            for_link = False
            for atom in interaction['atoms']:
                if atom[0] in '+-':
                    for_link = True
                    break
            if for_link:
                link.interactions[name].append(interaction)
                relevant_atoms.update(interaction['atoms'])
            else:
                block.interactions[name].append(interaction)

    # Prune the link to keep only the edges and particles that are
    # relevant.
    nodes = set(link.nodes())
    link.remove_nodes_from(nodes - relevant_atoms)

    # Revert the interactions back to regular dicts to avoid creating
    # keys when querying them.
    block.interactions = dict(block.interactions)
    link.interactions = dict(link.interactions)

    return block, link


def _clean_lines(lines):
    # TODO: merge continuation lines
    for line in lines:
        splitted = line.split(';', 1)
        if splitted[0].strip():
            yield splitted[0]


def _dihedral_sorted_center(atoms):
    #return sorted(atoms[1:-1])
    return atoms[1:-1]


def read_rtp(lines):
    _subsection_parsers = {
        'atoms': _atoms,
        'bonds': _base_rtp_parser('bonds', natoms=2),
        'angles': _base_rtp_parser('angles', natoms=3),
        'impropers': _base_rtp_parser('impropers', natoms=4),
        'cmap': _base_rtp_parser('cmap', natoms=5),
    }
    # An RTP file contains both the blocks and the links.
    # We first read everything in "pre-blocks"; then we separate the
    # blocks from the links.
    pre_blocks = {}
    bondedtypes = None
    cleaned = _clean_lines(lines)
    for section_name, section in _IterRTPSections(cleaned):
        if section_name == 'bondedtypes':
            bondedtypes = _parse_bondedtypes(section)
            continue
        block = Block()
        pre_blocks[section_name] = block
        block.name = section_name

        for subsection_name, subsection in section:
            if subsection_name in _subsection_parsers:
                _subsection_parsers[subsection_name](subsection, block)

    # Pre-blocks only contain the interactions that are explicitly
    # written in the file. Some are incomplete (missing implicit defaults)
    # or must be built from the "bondedtypes" rules.
    # If the "bondedtypes" rules are not defines (which should probably
    # not happen), then we only use what is explicitly written in the file.
    if bondedtypes is not None:
        for pre_block in pre_blocks.values():
            _complete_block(pre_block, bondedtypes)

    # At this point, the pre-blocks contain both the intra- and
    # inter-residues information. We need to split the pre-blocks into
    # blocks and links.
    blocks, links = _split_blocks_and_links(pre_blocks)
    return blocks, links
