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

from .ffinput import _tokenize, _parse_macro, _substitute_macros

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
        if not hasattr(obj, 'METH_DICT'):
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
    METH_DICT: dict[tuple[str], tuple[collections.abc.Callable, dict[str]]]
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
        """
        return line.startswith('[') and line.endswith(']')

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
