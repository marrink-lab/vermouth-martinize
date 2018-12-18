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
Contains the Mapping object and the associated parser.
"""
from collections import defaultdict, deque
from functools import partial

from .forcefield import FORCE_FIELDS
from .ffinput import _tokenize, _parse_atom_attributes, _parse_macro, _substitute_macros
from .graph_utils import MappingGraphMatcher
from .molecule import Block


from collections import defaultdict, deque
from functools import partial

from .log_helpers import StyleAdapter, get_logger
from .molecule import Block


LOGGER = StyleAdapter(get_logger(__name__))


def get_block(ff_name, type, resname):
    """
    Helper method that gets a a block associated with a name and a type from
    a specific :class:`vermouth.forcefield.ForceField`.

    Parameters
    ----------
    ff_name: str
        The name of the force field.
    type: str
        The type of block to get, e.g. "block" or "modification".
    resname: str
        The name of the block to get.

    Returns
    -------
    vermouth.molecule.Block or vermouth.molecule.Link
        The found block.
    """
    return getattr(FORCE_FIELDS[ff_name], type+'s')[resname]


class Mapping:
    """
    A mapping object that describes a mapping from one resolution to another.

    Attributes
    ----------
    blocks_from: networkx.Graph
        The graph which this :class:`Mapping` object can transform.
    blocks_to: vermouth.molecule.Block
        The :class:`vermouth.molecule.Block` we can transform to.
    references: collections.abc.Mapping
        A mapping of node keys in :attr:`blocks_to` to node keys in
        :attr:`blocks_from` that describes which node from should be taken as a
        reference when transferring node attributes.
    ff_from: vermouth.forcefield.ForceField
        The forcefield of :attr:`blocks_from`.
    ff_to: vermouth.forcefield.ForceField
        The forcefield of :attr:`blocks_to`.
    names: tuple[str]
        The names of the mapped blocks.
    mapping: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        The actual mapping that describes for every node key in
        :attr:`blocks_from` to what node key in :attr:`blocks_to` it
        contributes to with what weight.
        ``{node_from: {node_to: weight, ...}, ...}``.
    reverse_mapping: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        .. autoattribute:: reverse_mapping

    Note
    ----
    Only nodes described in :attr:`mapping` will be used.

    Parameters
    ----------
    block_from: networkx.Graph
        As per :attr:`blocks_from`.
    block_to: vermouth.molecule.Block
        As per :attr:`blocks_to`.
    mapping: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        As per :attr:`mapping`.
    references: collections.abc.Mapping
        As per :attr:`references`.
    ff_from: vermouth.forcefield.ForceField
        As per :attr:`ff_from`.
    ff_to: vermouth.forcefield.ForceField
        As per :attr:`ff_to`.
    extra: tuple
        Extra information to be attached to :attr:`blocks_to`.
    normalize_weights: bool
        Whether the weights should be normalized such that the sum of the
        weights of nodes mapping to something is 1.
    names: tuple
        As per :attr:`names`.
    """
    def __init__(self, block_from, block_to, mapping, references,
                 ff_from=None, ff_to=None, extra=(), normalize_weights=False,
                 names=tuple()):
        self.blocks_from = block_from
        self.blocks_to = block_to
        self.blocks_to.extra = extra
        self.references = references
        self.ff_from = ff_from
        self.ff_to = ff_to
        self.names = names
        self.mapping = mapping
        # Remove nodes not mapped from blocks_from
        unmapped = set(self.blocks_from.nodes.keys()) - set(self.mapping.keys())
        self.blocks_from.remove_nodes_from(unmapped)

        # Normalize the weights
        if normalize_weights:
            self._normalize_weights()

    @property
    def reverse_mapping(self):
        """
        The reverse of :attr:`mapping`.
        ``{node_to: {node_from: weight, ...}, ...}``
        """
        rev_mapping = defaultdict(dict)  # {idx to: {idx from: weight}}
        for idx_from in self.mapping:
            for idx_to, weight in self.mapping[idx_from].items():
                rev_mapping[idx_to][idx_from] = weight
        return dict(rev_mapping)

    def map(self, graph, node_match=None, edge_match=None):
        """
        Performs the partial mapping described by this object on `graph`. It
        first find the induced subgraph isomorphisms between `graph` and
        :attr:`blocks_from`, after which it will process the found isomorphisms
        according to :attr:`mapping`.

        None of the yielded dictionaries will refer to node keys of
        :attr:`blocks_from`. Instead, those will be translated to node keys of
        `graph` based on the found isomorphisms.

        Note
        ----
        Only nodes described in :attr:`mapping` will be used in the
        isomorphism.

        Parameters
        ----------
        graph: networkx.Graph
            The graph on which this partial mapping should be applied.
        node_match: collections.abc.Callable
            A function that should take two dictionaries with node attributes,
            and return `True` if those nodes should be considered equal, and
            `False` otherwise.
        edge_match: collections.abc.Callable
            A function that should take six arguments: two graphs, and four
            node keys. The first two node keys will be in the first graph and
            share an edge; and the last two node keys will be in the second
            graph and share an edge. Should return `True` if a pair of edges
            should be considered equal, and `False` otherwise.

        Yields
        ------
        tuple[dict, vermouth.molecule.Block, dict]
            A tuple containing 1) the correspondence between nodes in `graph`
            and nodes in :attr:`blocks_to`, with the associated weights; 2)
            :attr:`blocks_to`; and 3) :attr:`references` on which
            :attr:`mapping` has been applied.
        """
        if node_match is None:
            def node_match(node1, node2):
                return True

        if edge_match is None:
            def edge_match(node11, node12, node21, node22):
                return True
        else:
            edge_match = partial(edge_match, graph, self.blocks_from)

        return self._graph_map(graph, node_match, edge_match)

    def _graph_map(self, graph, node_match, edge_match):
        """
        Performs the partial mapping described in :meth:`map` using a
        "classical" subgraph isomorphism algorithm.
        """
        # 1 Find subgraph isomorphism between blocks_from and graph
        # 2 Translate found match ({graph idx: blocks from idx}) to indices in
        #   blocks_to using self.mapping
        # 3 Return found matches and blocks_to?
        # PS. I don't really like this, because this object is becoming too
        # intelligent by also having to do the isomorphism. On the other hand,
        # it makes sense from the maths point of view.
        matcher = MappingGraphMatcher(graph, self.blocks_from,
                                      node_match=node_match,
                                      edge_match=edge_match)
        for match in matcher.subgraph_isomorphisms_iter():
            rev_match = {v: k for k, v in match.items()}

            new_match = defaultdict(dict)
            for graph_idx, from_idx in match.items():
                new_match[graph_idx].update(self.mapping[from_idx])
            references = {out_idx: rev_match[ref_idx]
                          for out_idx, ref_idx in self.references.items()}
            yield dict(new_match), self.blocks_to, references

    def _normalize_weights(self):
        """
        Normalize weights in :attr:`mapping` such that the sum of the weights
        of nodes mapping to something is 1.
        """
        rev_mapping = self.reverse_mapping
        for idx_from in self.mapping:
            for idx_to, weight in self.mapping[idx_from].items():
                self.mapping[idx_from][idx_to] = weight/sum(rev_mapping[idx_to].values())


class MappingBuilder:
    """
    An object that is in charge of building the arguments needed to create a
    :class:`Mapping` object. It's attributes describe the information
    accumulated so far.

    Attributes
    ----------
    mapping: collections.defaultdict
    blocks_from: None or vermouth.molecule.Block
    blocks_to: None or vermouth.molecule.Block
    ff_from: None or vermouth.forcefield.ForceField
    ff_to: None or vermouth.forcefield.ForceField
    names: list
    references: dict
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the object to a clean initial state.
        """
        self.mapping = defaultdict(dict)
        self.blocks_from = None
        self.blocks_to = None
        self.ff_from = None
        self.ff_to = None
        self.names = []
        self.references = {}

    def to_ff(self, ff_name):
        """
        Sets :attr:`ff_to`

        Parameters
        ----------
        ff_name
        """
        self.ff_to = ff_name

    def from_ff(self, ff_name):
        """
        Sets :attr:`ff_from`

        Parameters
        ----------
        ff_name
        """
        self.ff_from = ff_name

    @staticmethod
    def _add_block(current_block, new_block):
        """
        Helper method that adds `new_block` to `current_block`, if the latteris
        not `None`. Otherwise, create a new block from `new_block`.

        Parameters
        ----------
        current_block: None or vermouth.molecule.Block
        new_block: vermouth.molecule.Block

        Returns
        -------
        vermouth.molecule.Block
            The combination of `current_block` and `new_block`
        """
        if current_block is None:
            current_block = new_block.to_molecule(default_attributes={})
        else:
            current_block.merge_molecule(new_block)
        return current_block

    def add_block_from(self, block):
        """
        Add a block to :attr:`blocks_from`. In addition, apply any 'replace'
        operation described by nodes on themselves::

            {'atomname': 'C', 'charge': 0, 'replace': {'charge': -1}}

        becomes::

            {'atomname': 'C', 'charge': -1}

        Parameters
        ----------
        block: vermouth.molecule.Block
            The block to add.
        """
        block = block.copy()
        for node in block.nodes.values():
            if 'replace' in node:
                node.update(node['replace'])
                del node['replace']
        self.blocks_from = self._add_block(self.blocks_from, block)

    def add_block_to(self, block):
        """
        Add a block to :attr:`blocks_to`.

        Parameters
        ----------
        block: vermouth.molecule.Block
            The block to add.
        """
        self.blocks_to = self._add_block(self.blocks_to, block)

    def add_node_from(self, attrs):
        """
        Add a single node to :attr:`blocks_from`.

        Parameters
        ----------
        attrs: dict[str]
            The attributes the new node should have.
        """
        if self.blocks_from is None:
            self.blocks_from = Block()
            idx = 0
        else:
            idx = max(self.blocks_from.nodes) + 1
        self.blocks_from.add_node(idx, **attrs)

    def add_node_to(self, attrs):
        """
        Add a single node to :attr:`blocks_to`.

        Parameters
        ----------
        attrs: dict[str]
            The attributes the new node should have.
        """
        if self.blocks_to is None:
            self.blocks_to = Block()
            idx = 1
        else:
            idx = max(self.blocks_to.nodes) + 1
        self.blocks_to.add_node(idx, **attrs)

    def add_edge_from(self, attrs1, attrs2):
        """
        Add a single edge to :attr:`blocks_from` between two nodes in
        :attr:`blocks_from` described by `attrs1` and `attrs2`. The nodes
        described should not be the same.

        Parameters
        ----------
        attrs1: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_from`
        attrs2: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_from`
        """
        nodes1 = list(self.blocks_from.find_atoms(**attrs1))
        nodes2 = list(self.blocks_from.find_atoms(**attrs2))
        assert len(nodes1) == len(nodes2) == 1
        assert nodes1 != nodes2
        self.blocks_from.add_edge(nodes1[0], nodes2[0])

    def add_edge_to(self, attrs1, attrs2):
        """
        Add a single edge to :attr:`blocks_to` between two nodes in
        :attr:`blocks_to` described by `attrs1` and `attrs2`. The nodes
        described should not be the same.

        Parameters
        ----------
        attrs1: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_to`
        attrs2: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_to`
        """
        nodes1 = list(self.blocks_to.find_atoms(**attrs1))
        nodes2 = list(self.blocks_to.find_atoms(**attrs2))
        assert len(nodes1) == len(nodes2) == 1
        assert nodes1 != nodes2
        self.blocks_to.add_edge(nodes1[0], nodes2[0])

    def add_mapping(self, attrs_from, attrs_to, weight):
        """
        Add part of a mapping to :attr:`mapping`. `attrs_from` uniquely
        describes a node in :attr:`blocks_from` and `attrs_to` a node in
        :attr:`blocks_to`. Adds a mapping between those nodes with the given
        `weight`.

        Parameters
        ----------
        attrs_from: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_from`
        attrs_to: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_to`
        weight: float
            The weight associated with this partial mapping.
        """
        nodes_from = list(self.blocks_from.find_atoms(**attrs_from))
        nodes_to = list(self.blocks_to.find_atoms(**attrs_to))
        assert len(nodes_from) == len(nodes_to) == 1
        self.mapping[nodes_from[0]][nodes_to[0]] = weight

    def add_reference(self, attrs_to, attrs_from):
        """
        Add a reference to :attr:`references`.

        Parameters
        ----------
        attrs_to: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_to`
        attrs_from: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_from`
        """
        nodes_to = list(self.blocks_to.find_atoms(**attrs_to))
        assert len(nodes_to) == 1
        node_to = nodes_to[0]
        nodes_from = set(self.blocks_from.find_atoms(**attrs_from))
        mapped_nodes = {from_ for from_ in self.mapping if node_to in self.mapping[from_]}
        nodes_from = nodes_from.intersection(mapped_nodes)
        assert len(nodes_from) == 1
        self.references[node_to] = next(iter(nodes_from))

    def get_mapping(self):
        """
        Instantiate a :class:`Mapping` object with the information accumulated
        so far, and return it.

        Returns
        -------
        Mapping
            The mapping object made from the accumulated information.
        """
        if self.blocks_from is None:
            return None
        mapping = Mapping(self.blocks_from, self.blocks_to, dict(self.mapping),
                          self.references, ff_from=self.ff_from, ff_to=self.ff_to,
                          names=tuple(self.names))
        return mapping


class MappingDirector:
    """
    A director in charge of parsing the new mapping format. It constructs a new
    :class:`Mapping` object by calling methods of it's builder (default
    :class:`MappingBuilder`) with the correct arguments.

    Attributes
    ----------
    RESNAME_NUM_SEP: str
        The character that separates a resname from a resnumber in shorthand
        block formats.
    RESIDUE_ATOM_SET: str
        The character that separates a residue identifier from an atomname.
    MAP_TYPES: list[str]
        Headers that define a type of mapping, rather than a section. They
        signal the start of a new :class:`Mapping` object.
    builder
        The builder used to build the :class:`Mapping` object. By default
        :class:`MappingBuilder`.
    identifiers: dict[str, dict[str]]
        All known identifiers at this point. The key is the actual identifier,
        prefixed with either "to\_" or "from\_", and the values are the
        associated node attributes.
    section: str
        The name of the section currently being processed.
    map_type: str
        The type of mapping currently being processed. Must be one of
        :attr:MAP_TYPES.
    from_ff: str
        The name of the forcefield from which this mapping describes a
        transfomation.
    to_ff: str
        The name of the forcefield to which this mapping describes a
        transfomation.
    macros: dict[str, str]
        A dictionary of known macros.
    """
    RESNAME_NUM_SEP = '#'
    RESIDUE_ATOM_SEP = ':'
    MAP_TYPES = ['block', 'modification']

    def __init__(self, builder=None):
        if builder is None:
            self.builder = MappingBuilder()
        else:
            self.builder = builder
        self._reset_file()

    def _reset_mapping(self):
        """
        Reinitialize attributes for a new mapping.
        """
        self._current_id = {'from_': None, 'to_': None}
        self.identifiers = {}
        self.builder.reset()
        self.from_ff = None
        self.to_ff = None

    def parse(self, file_handle):
        """
        Parse the data in `file_handle`.

        file_handle: io.TextIOBase
            The data stream to parse.

        Yields
        ------
        Mapping
            The mappings described by the data.
        """
        for lineno, line in enumerate(file_handle, 1):
            # TODO split off comments
            line = line.strip()
            if not line:
                continue
            if self._is_section_header(line):
                new_mapping = self._header(line)
                if new_mapping:
                    mapping = self.builder.get_mapping()
                    if mapping is not None:
                        yield mapping
                    self._reset_mapping()
            else:
                self._parse_line(line, lineno)
        mapping = self.builder.get_mapping(self.map_type)
        if mapping is not None:
            yield mapping
        self._reset_mapping()

    @staticmethod
    def _is_section_header(line):
        """
        Parameters
        ----------
        line: str
            A line of text.

        Returns
        -------
        bool
            ``True`` iff `line` is a section header.
        """
        return line.startswith('[') and line.endswith(']')
    
    @staticmethod
    def _section_to_method(section):
        method_name = '_' + section.replace(' ', '_')
        return method_name

    @staticmethod
    def _section_to_method(section):
        """
        Translates `section` to a method name by replacing all spaces with
        underscores, and prefixes it with an underscore.

        Parameters
        ----------
        section: str
            A line of text.

        Returns
        -------
        str
            The translation of `section` to a method name.
        """
        method_name = '_' + section.replace(' ', '_')
        return method_name

    def _parse_line(self, line, lineno):
        """
        Parse `line` with line number `lineno`.

        Parameters
        ----------
        line: str
        lineno: int

        Returns
        -------
        None
        """
        line = _substitute_macros(line, self.macros)
        method_name = self._section_to_method(self.section)
        try:
            getattr(self, method_name)(line)
        except Exception:
            LOGGER.error("Problems parsing line {}. I think it should be a '{}'"
                         " line, but I can't parse it as such.",
                         lineno, self.section)
            raise

    def _header(self, line):
        """
        Parses a section header. Sets :attr:`section` and :attr:`map_type` when
        applicable. Does not check whether `line` is a valid section header.

        Parameters
        ----------
        line: str

        Returns
        -------
        bool
            ``True`` iff the header parsed is in :attr:`MAP_TYPES`.

        Raises
        ------
        KeyError
            If the section header is unknown.
        """
        section = line.strip('[ ]').casefold()
        method_name = self._section_to_method(section)
        try:
            if section not in self.MAP_TYPES:
                getattr(self, method_name)
        except AttributeError as err:
            raise KeyError('Section "{}" is unknown'.format(section)) from err
        else:
            self.section = section
            if section in self.MAP_TYPES:
                self.map_type = section
                return True
            else:
                return False

    def _parse_blocks(self, line):
        """
        Helper method for parsing to_blocks and from_blocks. It parses a line
        containing either a single longhand block description, or multiple
        shorthand block descriptions.

        Parameters
        ----------
        line: str

        yields
        ------
        tuple[str, dict[str]]
            A tuple if an identifier, and it's associated attributes.
        """
        tokens = list(_tokenize(line))
        if len(tokens) == 2 and tokens[1].startswith('{') and tokens[1].endswith('}'):
            # It's definitely full spec.
            identifier = tokens[0]
            attrs = _parse_atom_attributes(tokens[1])
            yield identifier, attrs
        else:  # It must be shorthand
            for identifier in tokens:
                resname, resid = self._parse_block_shorthand(identifier)
                attrs = {'resname': resname, 'resid': resid}
                yield identifier, attrs

    def _parse_block_shorthand(self, token):
        """
        Helper method for parsing block shorthand.

        Parameters
        ----------
        token: str

        Returns
        -------
        tuple[str, int]
            A tuple of a resname and a resid.
        """
        if self.RESNAME_NUM_SEP in token:
            # PO4#3
            resname, resid = token.split(self.RESNAME_NUM_SEP)
        elif token[-1].isdigit():
            # ALA2
            for idx, char in enumerate(reversed(token)):
                if not char.isdigit():
                    idx = len(token) - idx
                    resname = token[:idx]
                    resid = int(token[idx:])
                    break
        else:
            # ALA
            resname = token
            resid = 1
        return resname, resid

    def _resolve_atom_spec(self, atom_str, prefix=''):
        """
        Helper method that, given an atom token and a prefix ("to_" or "from_")
        will find the associated node attributes for that atom token. It will
        either separate the given identifier and atomname and look up the
        associated node attributes in :attr:`identifiers`; or take the last
        specified identifier.

        Parameters
        ----------
        atom_str: str
        prefix: str

        Returns
        -------
        dict[str]
            The node attributes that describe this atom.
        """
        if self.RESIDUE_ATOM_SEP in atom_str:
            id_, name = atom_str.split(self.RESIDUE_ATOM_SEP)
        else:
            id_, name = None, atom_str

        if id_ is None:
            options = {name for name in self.identifiers if name.startswith(prefix)}
            if len(options) == 1:
                id_ = next(iter(options))
                id_ = id_[len(prefix):]

        if id_ is None:
            attrs = self._current_id[prefix].copy()
        else:
            attrs = self.identifiers[prefix + id_].copy()
            self._current_id[prefix] = self.identifiers[prefix + id_]

        attrs['atomname'] = name
        if self.map_type == 'modification':
            del attrs['resname']
        return attrs

    def _to(self, line):
        """
        Parses a "to" section and sets :attr:`to_ff`.

        Parameters
        ----------
        line: str
        """
        self.to_ff = line
        self.builder.to_ff(self.to_ff)

    def _from(self, line):
        """
        Parses a "from" section and sets :attr:`from_ff`.

        Parameters
        ----------
        line: str
        """
        self.from_ff = line
        self.builder.from_ff(self.from_ff)

    def _from_blocks(self, line):
        """
        Parses a "from blocks" section and add to :attr:`identifiers`. Calls
        :method:`builder.add_block_from`.

        Parameters
        ----------
        line: str
        """
        for identifier, attrs in self._parse_blocks(line):
            if isinstance(attrs.get('resname'), str):
                block = get_block(self.from_ff, self.map_type, attrs['resname'])
                self.builder.add_block_from(block)
            self.identifiers['from_' + identifier] = attrs

    def _to_blocks(self, line):
        """
        Parses a "to blocks" section and add to :attr:`identifiers`. Calls
        :method:`builder.add_block_to`.

        Parameters
        ----------
        line: str
        """
        for identifier, attrs in self._parse_blocks(line):
            if isinstance(attrs.get('resname'), str):
                block = get_block(self.to_ff, self.map_type, attrs['resname'])
                self.builder.add_block_to(block)
            self.identifiers['to_' + identifier] = attrs

    def _from_nodes(self, line):
        """
        Parses a "from nodes" section. Calls :method:`builder.add_node_from`.

        Parameters
        ----------
        line: str
        """
        name, *attrs = _tokenize(line)
        if attrs:
            attrs = _parse_atom_attributes(*attrs)
        else:
            attrs = {}
        if 'atomname' not in attrs:
            attrs['atomname'] = name
        self.builder.add_node_from(attrs)

    def _to_nodes(self, line):
        """
        Parses a "to nodes" section. Calls :method:`builder.add_node_to`.

        Parameters
        ----------
        line: str
        """
        name, *attrs = _tokenize(line)

        if attrs:
            attrs = _parse_atom_attributes(*attrs)
        else:
            attrs = {}
        if 'atomname' not in attrs:
            attrs['atomname'] = name
        self.builder.add_node_to(attrs)

    def _from_edges(self, line):
        """
        Parses a "from edges" section. Calls :method:`builder.add_edge_from`.

        Parameters
        ----------
        line: str
        """
        at1, at2 = line.split()
        attrs1 = self._resolve_atom_spec(at1, 'from_')
        attrs2 = self._resolve_atom_spec(at2, 'from_')
        self.builder.add_edge_from(attrs1, attrs2)

    def _to_edges(self, line):
        """
        Parses a "to edges" section. Calls :method:`builder.add_edge_to`.

        Parameters
        ----------
        line: str
        """
        at1, at2 = line.split()
        attrs1 = self._resolve_atom_spec(at1, 'to_')
        attrs2 = self._resolve_atom_spec(at2, 'to_')
        self.builder.add_edge_to(attrs1, attrs2)

    def _mapping(self, line):
        """
        Parses a "mapping" section. Calls :method:`builder.add_mapping`.

        Parameters
        ----------
        line: str
        """
        from_, to_, *weight = line.split()
        if weight:
            weight = int(weight[0])
        else:
            weight = 1

        attrs_from = self._resolve_atom_spec(from_, 'from_')
        attrs_to = self._resolve_atom_spec(to_, 'to_')

        self.builder.add_mapping(attrs_from, attrs_to, weight)

    def _reference_atoms(self, line):
        """
        Parses a "reference atom" section. Calls
        :method:`builder.add_reference`.

        Parameters
        ----------
        line: str
        """
        to_, from_ = line.split()
        attrs_to = self._resolve_atom_spec(to_, 'to_')
        attrs_from = self._resolve_atom_spec(from_, 'from_')
        self.builder.add_reference(attrs_to, attrs_from)

    def _macros(self, line):
        """
        Parses a "macros" section. Adds to :attr:`macros`.

        Parameters
        ----------
        line: str
        """
        line = deque(_tokenize(line))
        _parse_macro(line, self.macros)

def parse_mapping_file(filepath):
    """
    Parses a mapping file.

    Parameters
    ----------
    filepath: str
        The path of the file to parse.

    Returns
    -------
    list[Mapping]
        A list of all mappings described in the file.
    """
    with open(filepath) as map_in:
        director = MappingDirector()
        mappings = list(director.parse(map_in))
    return mappings
