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
Helper functions for parsers
"""
from collections import deque

# This file contains helper methods and infrastructure for parsers. The class
# SectionLineParser in particular is a powerful tool that is intended to make
# parsing section based files easier. There's a fair chance it turned out to be
# a multi-tentacled Lovecraftion horror that's in charge of slightly magical
# switchboard. If it ever breaks I'm very sorry for you. Pray to your deity of
# choice and prepare your sacrificial chicken.


class SectionParser(type):
    """
    Metaclass (!) that populates the `METH_DICT` attribute of new classes. The
    contents of `METH_DICT` are set by reading the `_section_names` attribute
    of all its attributes. You can conveniently set `_section_names` attributes
    using the :meth:`section_parser` decorator.
    """
    def __new__(mcs, name, bases, attrs, **kwargs):
        obj = super().__new__(mcs, name, bases, attrs, **kwargs)
        #if not hasattr(obj, 'METH_DICT'):
        obj.METH_DICT = {}
        mapping = obj.METH_DICT

        for attribute_name in dir(obj):
            attribute = getattr(obj, attribute_name)
            try:
                section_names = attribute._section_names
            except AttributeError:
                pass
            else:
                for names, kwargs in section_names.items():
                    mapping[names] = (attribute, kwargs)
        return obj

    @staticmethod
    def section_parser(*names, **kwargs):
        """
        Parameters
        ----------
        names: tuple[collections.abc.Hashable]
            The section names that should be associated with the decorated
            function.
        kwargs: dict[str]
            The keyword arguments with which the decorated function should be
            called.
        """
        def wrapper(method):
            if not hasattr(method, '_section_names'):
                method._section_names = {}
            method._section_names[names] = kwargs
            return method
        return wrapper


class LineParser:
    """
    Class that describes a parser object that parses a file line by line.
    Subclasses will probably want to override the methods :meth:`dispatch`,
    :meth:`parse_line`, and/or :meth:`finalize`:

      - :meth:`dispatch` is called for every line and should return the
        function that should be used to parse that line.
      - :meth:`parse_line` is called by the default implementation of
        :meth:`dispatch` for every line.
      - :meth:`finalize` is called at the end of the file.
    """
    COMMENT_CHAR = '#'

    def parse(self, file_handle):
        """
        Reads lines from `file_handle`, and calls :meth:`dispatch` to find
        which method to call to do the actual parsing. Yields the result of
        that call, if it's not `None`.
        At the end, calls :meth:`finalize`, and yields its results, iff
        it's not None.

        Parameters
        ----------
        file_handle: collections.abc.Iterable[str]
            The data to parse. Should produce lines of data.

        Yields
        ------
        object
            The results of dispatching to parsing methods, and of
            :meth:`finalize`.
        """
        lineno = 0
        for lineno, line in enumerate(file_handle, 1):
            line, _ = split_comments(line, self.COMMENT_CHAR)
            if not line:
                continue
            result = self.dispatch(line)(line, lineno)

            if result is not None:
                yield result

        result = self.finalize(lineno)
        if result is not None:
            yield result

    def finalize(self, lineno=0):
        """
        Wraps up. Is called at the end of the file.
        """
        return

    def dispatch(self, line):
        """
        Finds the correct method to parse `line`. Always returns
        :meth:`parse_line`.
        """
        return self.parse_line

    def parse_line(self, line, lineno):
        """
        Does nothing and should be overridden by subclasses.
        """
        return


class SectionLineParser(LineParser, metaclass=SectionParser):
    """
    Baseclass for all parsers that have to parse file formats that are based on
    sections. Parses the `macros` section.
    Subclasses will probably want to override :meth:`finalize` and/or
    :meth:`finalize_section`.

    :meth:`finalize_section` is called with the previous section whenever a
    section ends.

    Attributes
    ----------
    section: list[str]
        The current section.
    macros: dict[str, str]
        A set of subsitution rules as parsed from a `macros` section.
    """
    METH_DICT = {}
    """
    A dict of all known parser methods, mapping section names to the
    function to be called and the associated keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        self.macros = {}
        self.section = []
        super().__init__(*args, **kwargs)

    def dispatch(self, line):
        """
        Looks at `line` to see what kind of line it is, and returns either
        :meth:`parse_header` if `line` is a section header or
        :meth:`parse_section` otherwise. Calls :meth:`is_section_header` to see
        whether `line` is a section header or not.

        Parameters
        ----------
        line: str

        Returns
        -------
        collections.abc.Callable
            The method that should be used to parse `line`.
        """
        if self.is_section_header(line):
            return self.parse_header
        else:
            return self.parse_section

    def finalize(self, lineno=0):
        """
        Called after the last line has been parsed to wrap up. Resets
        the instance and calls :meth:`finalize_section`.

        Arguments
        ---------
        lineno: int
            The line number.
        """
        prev_section = self.section
        self.section = []
        result = self.finalize_section(prev_section, prev_section)
        self.macros = {}
        self.section = None
        return result

    def finalize_section(self, previous_section, ended_section):
        """
        Called once a section is finished. Currently does nothing.

        Arguments
        ---------
        previous_section: list[str]
            The last parsed section.
        ended_section: list[str]
            The sections that have been ended.
        """
        return

    def parse_section(self, line, lineno):
        """
        Parse `line` with line number `lineno` by looking up the section in
        :attr:`METH_DICT` and calling that method.

        Parameters
        ----------
        line: str
        lineno: int

        Returns
        -------
        object
            The result returned by calling the registered method.
        """
        line = _substitute_macros(line, self.macros)
        if tuple(self.section) not in self.METH_DICT:
            raise IOError("Can't parse line {} in section '{}' because the "
                          "section is unknown".format(lineno, self.section))
        try:
            method, kwargs = self.METH_DICT[tuple(self.section)]
            return method(self, line, lineno, **kwargs)
        except Exception as error:
            raise IOError("Problems parsing line {}. I think it should be a "
                          "'{}' line, but I can't parse it as such."
                          "".format(lineno, self.section)) from error

    def parse_header(self, line, lineno=0):
        """
        Parses a section header with line number `lineno`. Sets :attr:`section`
        when applicable. Does not check whether `line` is a valid section
        header.

        Parameters
        ----------
        line: str
        lineno: str

        Returns
        -------
        object
            The result of calling :meth:`finalize_section`, which is called
            if a section ends.

        Raises
        ------
        KeyError
            If the section header is unknown.
        """
        prev_section = self.section

        section = self.section + [line.strip('[ ]').casefold()]

        ended = []
        while tuple(section) not in self.METH_DICT and len(section) > 1:
            ended.append(section.pop(-2))  # [a, b, c, d] -> [a, b, d]

        self.section = section
        if prev_section:
            result = self.finalize_section(prev_section, ended)
            return result

    @staticmethod
    def is_section_header(line):
        """
        Parameters
        ----------
        line: str
            A line of text.

        Returns
        -------
        bool
            ``True`` iff `line` is a section header.

        Raises
        ------
        IOError
            The line starts like a section header but looks misformatted.
        """
        if line.startswith('['):
            if line.endswith(']'):
                return True
            else:
                raise IOError('Section header looks misformatted.')
        return False

    @SectionParser.section_parser('macros')
    def _macros(self, line, lineno=0):
        """
        Parses a "macros" section. Adds to :attr:`macros`.

        Parameters
        ----------
        line: str
        """
        line = deque(_tokenize(line))
        _parse_macro(line, self.macros)


def split_comments(line, comment_char=';'):
    """
    Splits `line` at the first occurence of `comment_char`.

    Parameters
    ----------
    line: str
    comment_char: str

    Returns
    -------
    tuple[str, str]
        `line` before and after `comment_char`, respectively. If `line` does
        not contain `comment_char`, the second element will be an empty string.
    """
    split = line.split(comment_char, 1)
    data = split[0].strip()
    if len(split) == 1:
        return data, ''
    else:
        return data, split[1].strip()


def _tokenize(line):
    """
    Split an interaction line into its elementary components.

    An interaction line is any uncommented and non empty line that follows a
    section header about an interaction type. Such a line is composed of the
    following parts:

    * a list of atoms involved in the interaction,
    * an optional delimiter that indicates the end of the atom list,
    * a list of parameters for the interaction.

    The list of atoms is *a minima* a list of atom references. In blocks, these
    references can be atom 1-based indices referring to the order of the atoms
    in the "[ atoms ]" section. It is however more readable, and more robust,
    to refer to atoms by their name. Only the reference by name is allowed in
    links, as links may not have a full "[ atoms ]" section. In links, each
    atom reference can be complemented by atom attributes to specify the  scope
    of the link. These attribute follow the atom reference and are formatted
    like a python dictionary.

    The end-of-atoms delimiter is useful for interaction types that are not
    explicitly encoded in the parser. It allows to indicate when the list of
    atoms ends, and where the list of parameters starts. Two dashes ("--") are
    used as the delimiter. The delimiter is optional for the interaction types
    that are explicitly encoded in the parser and that refer to a fixed number
    of atoms.

    The list of parameters will be copied as-is in an ITP file.

    In its simplest form, an interaction line is what is used in an ITP file.
    Here is an example for a bond:

        2  3  1  0.2  1000

    The two first numbers refer to the second and third atoms of the block,
    respectively. The next three values are the parameters for a bond (*i.e.*
    the function type, the equilibrium distance, and the force constant).

    The two first numbers could be replaced by the corresponding atom names:

        PO4  GL1  1 0.2  1000

    where "PO4" and "GL1" are the names of the second and third atoms of the
    block.

    Optionally, the "--" delimiter can be used after the list of atoms:

        PO4  GL1  --  1 0.2  1000

    If the line is part of a link, then the atom selection may be limited in
    scope. Atom attributes is how to implement such scope limitation:

        BB {'resname': 'ALA', 'secstruc': 'H'} BB {'resname': 'LYS', 'secstruc': 'H', 'order': +1} 1 0.2 1000

    Here, we add a bond to the current link. At one end of the bond is the atom
    named "BB" and annotated as part of an alpha helix ('secstruc': 'H') of a
    residue called "ALA". On the other end of the link is an other
    atom named "BB" that is part of an alpha helix, but that is part of the
    next residue ('order': +1) if this next residue is named "LYS".

    The order parameter has a shortcut in the form of a + or - prefix to the
    atom reference name. Then, "+ATOM" refers to "ATOM" in the next residue,
    and is equivalent to "ATOM {'order': +1}"; "-ATOM" refers to the previous
    residue. There can be multiple + or -, "++ATOM" is equivalent to "ATOM
    {'order': +2}".

    When using attributes, the optional delimiter can increase the readability:

        BB {'resname': 'ALA', 'secstruc': 'H'} +BB {'resname': 'LYS', 'secstruc': 'H'} -- 1 0.2 1000

    Tokens on an interaction line are its different elements. These elements
    are considered as one token each: am atom reference, a set of atom
    attributes, the optional delimiter, each space-separated element of the
    parameter list. The line above splits into the following tokens:

    * ``BB``
    * ``{'resname': 'ALA', 'secstruc': 'H'}``
    * ``+BB``
    * ``{'resname': 'LYS', 'secstruc': 'H'}``
    * ``--``
    * ``1``
    * ``0.2``
    * ``1000``

    Atom attributes can be written next to the previous or the next token
    without an explicit separator. The two following lines yield the same three
    tokens:

        ATOM1{attributes}ATOM2
        ATOM1 {attributes} ATOM2

    Parameters
    ----------
    line: str

    Returns
    -------
    list of str
    """
    separators = ' \t\n'
    tokens = []
    start = 0
    end = -1
    # Find the first non-separator character
    for start, char in enumerate(line):
        if char not in separators:
            break

    # Find the tokens. This has to be a while-loop because we cannot predict
    # what will be the next value of start.
    while start < len(line):
        end = start
        # We count the brackets because if a token starts with an opening
        # bracket, we want to end it with the *matching* closing bracket.
        # Note also that we do not yet implement a way to escape a bracket, nor
        # do we check if the bracket is not part of a string.
        brackets = 0
        for end, end_char in enumerate(line[start:], start=start):
            if end_char == '{':
                # We reached an opening bracket. If it is the first character
                # of the token or if we are already engaged in a bracketized
                # token, then we go on. But if the current token was not
                # a bracketized token, it means we are at the beginning of
                # a new token, so we treat the opening bracket as a separator.
                if not brackets and end != start:
                    end -= 1
                    break
                brackets += 1
            elif end_char == '}':
                brackets -= 1
                if not brackets:
                    break
            elif end_char in separators:
                if not brackets:
                    # We reached a separator. We do not want the separator to
                    # be included in the token, so we push the end by one
                    # character to the left.
                    end -= 1
                    break
        if brackets > 0:
            msg = 'Unexpected end of line. A closing bracket is missing.'
            raise IOError(msg)
        elif brackets < 0:
            msg = 'An opening bracket is missing.'
            raise IOError(msg)

        token = line[start:end + 1]
        if token:
            tokens.append(token)

        # Find the beginning of the next token.
        start = end + 1
        while start < len(line) and line[start] in separators:
            start += 1
    return tokens


def _substitute_macros(line, macros):
    r"""
    Substitute macros by their content.

    A macro starts with a '$' and ends with one amongst ' ${}\n\t"'.

    Parameters
    ----------
    line: str
        The line to fix.
    macros: dict[str, str]
        Keys are macro names, values are the replacement content.

    Returns
    -------
    str
    """
    start = None
    while True:  # stops when start < 0
        start = line.find('$', start)
        if start < 0:
            break
        for end, char in enumerate(line[start + 1:], start=start + 1):
            if char in ' \t\n{}$"':
                break
        else: # no break
            end += 1
        macro_name = line[start + 1:end]
        macro_value = macros[macro_name]
        line = line[:start] + macro_value + line[end:]
        end = start + len(macro_value)
    return line


def _parse_macro(tokens, macros):
    if len(tokens) > 2:
        raise IOError('Unexpected column in macro definition.')
    elif len(tokens) < 2:
        raise IOError('Missing column in macro definition.')
    macro_name = tokens.popleft()
    macro_value = tokens.popleft()
    macros[macro_name] = macro_value
