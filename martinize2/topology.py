import collections
import networkx as nx

# Name of the subsections in RTP files.
# Names starting with a '_' are for internal use.
RTP_SUBSECTIONS = ('atoms', 'bonds', 'angles', 'dihedrals',
                   'impropers', 'cmap', 'exclusions',
                   '_bondedtypes')


class Block(object):
    """
    Residue topology template

    Attributes
    ----------

    name: str or None
        The name of the residue. Set to `None` if undefined.
    atoms: list of dict
        The atoms in the residue. Each atom is a dict with *a minima* a key
        'name' for the name of the atom, and a key 'atype' for the atom type.
        An atom can also have a key 'charge', 'charge_group', 'comment', or any
        arbitrary key.
    interactions: dict
        All the known interactions. Each item of the dictionary is a type of
        interaction, with the key being the name of the kind of interaction
        using Gromacs itp/rtp conventions ('bonds', 'angles', ...) and the
        value being a list of all the interactions of that type in the residue.
        An interaction is a dict with a key 'atoms' under which is stored the
        list of the atomsi involved (refered by their name), a key 'parameters'
        under which is stored an arbitrary list of non-atom parameters as
        written in a RTP file, and arbitrary keys to store custom metadata. A
        given interaction can have a comment under the key 'comment'.
    """
    def __init__(self):
        self.name = None
        self.atoms = []
        self.interactions = {}

    def __repr__(self):
        name = self.name
        if name is None:
            name = 'Unnamed'
        return '<{} "{}" at 0x{:x}>'.format(self.__class__.__name__,
                                          name, id(self))


class Link(nx.Graph):
    """
    Template link between two residues.
    """
    def __init__(self):
        super(Link, self).__init__(self)
        self.interactions = {}


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
        raise RuntimeError('I am almost sure I should not be here...')

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
        raise RuntimeError('Hum... There is a bug in the RTP reader.')

    def __iter__(self):
        return self


class RTPReader(object):
    def __init__(self):
        self._subsection_parsers = {
            'atoms': self._atoms,
            'bonds': self._base_rtp_parser('bonds', natoms=2),
            'angles': self._base_rtp_parser('angles', natoms=3),
            'impropers': self._base_rtp_parser('impropers', natoms=4),
            'cmap': self._base_rtp_parser('cmap', natoms=5),
        }

    def read_rtp(self, lines):
        # An RTP file contains both the blocks and the links.
        # We first read everything in "pre-blocks"; then we separate the
        # blocks from the links.
        pre_blocks = {}
        cleaned = _clean_lines(lines)
        for section_name, section in _IterRTPSections(cleaned):
            if section_name == 'bondedtypes':
                # I'll implement the support for the bonded type
                # section latter.
                continue
            block = Block()
            pre_blocks[section_name] = block
            block.name = section_name

            for subsection_name, subsection in section:
                if subsection_name in self._subsection_parsers:
                    self._subsection_parsers[subsection_name](subsection, block)

        # At this point, the pre-blocks contain both the intra- and
        # inter-residues information. We need to split the pre-blocks into
        # blocks and links.
        blocks, links = self._split_blocks_and_links(pre_blocks)
        return blocks, links

    @staticmethod
    def _atoms(subsection, block):
        for line in subsection:
            name, atype, charge, charge_group = line.split()
            atom = {
                'name': name,
                'atype': atype,
                'charge': float(charge),
                'charge_group': int(charge_group),
            }
            block.atoms.append(atom)

    @staticmethod
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

    def _split_blocks_and_links(self, pre_blocks):
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
            block, link = self._split_block_and_link(pre_block)
            blocks[name] = block
            links.append(link)
        return blocks, links

    @staticmethod
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
                block.atoms.append(atom)
        
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


# There is no need for more than one instance of RTPReader. The class is only
# there to group all the relevant functions.
read_rtp = RTPReader().read_rtp
