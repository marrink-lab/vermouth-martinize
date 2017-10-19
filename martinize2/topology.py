import collections


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


class Link(object):
    pass


class _IterRTPSubsectionLines(object):
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
        if line[0] == '[':
            self.parent.buffer.append(line)
            self.running = False
            raise StopIteration
        stripped = line.strip()
        if stripped[0] == '[':
            name = stripped[1:-1].strip()
            subsection = _IterRTPSubsectionLines(self)
            self.current_subsection = subsection
            return name, subsection

    def __iter__(self):
        return self

    def flush(self):
        for line in self:
            pass


class _IterRTPSections(object):
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
        if line[0] == '[':
            name = line.strip()[1:-1].strip()
            section = _IterRTPSubsections(self)
            self.current_section = section
            # The "[ bondedtypes ]" is special in the sense that it does
            # not have subsection, but instead have its content directly
            # under it. This breaks the neat hierarchy the rest of the file
            # has. Here we restore the hierarchy by faking that the file
            # contains:
            #
            #    [ bondedtypes ]
            #     [ bondedtypes ]
            #
            # For now, I do that on the basis of the section name. If other
            # sections are special that way, I'll detect them with a look
            # ahead.
            if name == 'bondedtypes':
                section.buffer.append(' [ bondedtypes ]')
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
            'improper': self._base_rtp_parser('impropers', natoms=4),
            # cmap?
        }

    def read_rtp(self, lines):
        blocks = {}
        links = []
        cleaned = _clean_lines(lines)
        for section_name, section in _IterRTPSections(cleaned):
            if section_name == 'bondedtypes':
                # I'll implement the support for the bonded type
                # section latter.
                continue
            block = Block()
            blocks[section_name] = block
            block.name = section_name

            for subsection_name, subsection in section:
                if subsection_name in self._subsection_parsers:
                    self._subsection_parsers[subsection_name](subsection, block)
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
                # For now we ignore the links
                skip = False
                for atom in atoms:
                    if atom[0] in '+-':
                        skip=True
                if skip:
                    continue
                parameters = splitted[natoms:]
                interactions.append({'atoms': atoms, 'parameters': parameters})
            block.interactions[interaction_name] = interactions
        return wrapped


def _clean_lines(lines):
    # TODO: merge continuation lines
    for line in lines:
        splitted = line.split(';', 1)
        if splitted[0].strip():
            yield splitted[0]


# There is no need for more than one instance of RTPReader. The class is only
# there to group all the relevant functions.
read_rtp = RTPReader().read_rtp
