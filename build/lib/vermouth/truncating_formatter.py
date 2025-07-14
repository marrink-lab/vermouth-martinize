#!/usr/bin/env python3
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
Provides a string formatter that can not only pad strings to a specified
length if they're too short, but also truncate them if they're too long.
"""


import string
import re
from collections import namedtuple

FormatSpec = namedtuple('FormatSpec', 'fill align sign alt zero_padding width comma decimal precision type')


class TruncFormatter(string.Formatter):
    """
    Adds the 't' option to the format specification mini-language at the end of
    the format string. If provided, the produced formatted string will be
    truncated to the specified length.
    """
    # https://stackoverflow.com/questions/44551535/access-the-cpython-string-format-specification-mini-language-parser
    format_spec_re = r'(([\s\S])?([<>=\^]))?([\+\- ])?(#)?(0)?(\d*)?(,)?((\.)(\d*))?([sbcdoxXneEfFgGn%])?'
    format_spec_re = re.compile(format_spec_re)

    def format_field(self, value, format_spec):
        """
        Implements the 't' option to truncate strings that are too long to the
        required width.

        Parameters
        ----------
        value
            The object to format.
        format_spec: str
            The format_spec describing how `value` should be formatted

        Returns
        str
            `value` formatted as per `format_spec`
        """
        if format_spec.endswith('t'):
            truncate = True
            format_spec = format_spec[:-1]
        else:
            truncate = False
        result = super().format_field(value, format_spec)
        # From here on we know the format spec is valid
        spec = FormatSpec(*self.format_spec_re.fullmatch(format_spec).group(2, 3, 4, 5, 6, 7, 8, 10, 11, 12))
        if spec.width:
            spec = spec._replace(width=int(spec.width))
        else:
            spec = spec._replace(width=0)
        if not truncate or spec.width == 0 or len(result) <= spec.width:
            return result
        # skip groups not interested in
        if not spec.type:
            if isinstance(value, str):
                spec = spec._replace(type='s')
            elif isinstance(value, int):
                spec = spec._replace(type='d')
            elif isinstance(value, float):
                spec = spec._replace(type='g')

        if not spec.align:
            if spec.type in 's':
                spec = spec._replace(align='<')
            elif spec.type in 'bcdoxXn' or spec.type in 'eEfFgGn%':
                spec = spec._replace(align='>')

        # We know len(result) > width. So there's no fill characters.
        # We also have at least width, type and align at this point.
        # We should probably do something special when it's a number with a
        # magic formatting prefix (0b, 0o, 0x) or if it has a sign. Idem for
        # exponent notation. Maybe, for numerical types we should round instead
        # of truncate the string.
        overflow = len(result) - spec.width
        if spec.align == '<':  # left chars most significant. e.g. str
            result = result[:-overflow]
        elif spec.align == '>':  # right characters most significant. e.g. int
            result = result[overflow:]
        elif spec.align == '=':  # padding between sign and digits +0000120
            # Note that this is the default for fill character 0
            raise NotImplementedError
        elif spec.align == '^':  # centered
            result = result[overflow//2:-overflow//2]

        return result


# if __name__ == '__main__':
#     formatter = TruncFormatter()
#
#     str_data = 'abcde'
#
#     print(formatter.format('{}', str_data))
#
#     print(formatter.format('"{:4.4}"', str_data))
#     print(formatter.format('"{:>4.4}"', str_data))
#     print(formatter.format('"{:4t}"', str_data))
#     print(formatter.format('"{:>4t}"', str_data))
#     print(formatter.format('"{:5t}"', str_data))
#     print(formatter.format('"{:6t}"', str_data))
#
#     int_data = 123456789
#
#     print(formatter.format('"{:4}"', int_data))
#     print(formatter.format('"{:>4}"', int_data))
#     print(formatter.format('"{:4t}"', int_data))
#     print(formatter.format('"{:>4t}"', int_data))
#     print(formatter.format('"{:<4t}"', int_data))
#     print(formatter.format('"{:^4t}"', int_data))
#     print(formatter.format('"{:5t}"', int_data))
#     print(formatter.format('"{:6t}"', int_data))
#
#     print(formatter.format('"{:11t}"', int_data))
#     print(formatter.format('"{:<11t}"', int_data))
#     print(formatter.format('"{:^11t}"', int_data))
