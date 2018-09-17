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
Provide some helper classes to allow new style brace formatting for logging and
processing the `type` keyword.
"""
import logging


class Message:
    """
    Class that defers string formatting until it's ``__str__`` method is
    called.
    """
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        return self.fmt.format(*self.args)

    def __repr__(self):
        return '{}.format(*{})'.format(self.fmt, self.args)


class StyleAdapter(logging.LoggerAdapter):
    """
    Logging adapter that encapsulate messages in :class:`Message`, allowing
    ``{}`` style formatting.
    """
    def __init__(self, logger, extra=None):
        if extra is None:
            extra = {}
        super().__init__(logger, extra)

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, Message(msg, args), **kwargs)

    def __getattr__(self, item):
        # Delegate attribute lookup further down the chain of LoggerAdapters
        return getattr(self.logger, item)


class TypeAdapter(logging.LoggerAdapter):
    """
    Logging adapter that takes the `type` keyword argument passed to logging
    calls and passes adds it to the `extra` attribute.

    Parameters
    ----------
    logger: logging.Logger or logging.LoggerAdapter
        As described in :class:`logging.LoggerAdapter`.
    extra: dict
        As described in :class:`logging.LoggerAdapter`.
    default_type: str
        The type of the messages if none is given.
    """
    def __init__(self, logger, extra=None, default_type='general'):
        if extra is None:
            extra = {}
        self.default_type = default_type
        super().__init__(logger, extra)

    def log(self, level, msg, *args, type=None, **kwargs):  # pylint: disable=arguments-differ
        if type is None:
            type = self.default_type
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            kwargs['extra'].update(type=type)
            self.logger.log(level, msg, *args, **kwargs)

    def __getattr__(self, item):
        # Delegate attribute lookup further down the chain of LoggerAdapters
        return getattr(self.logger, item)


def get_logger(name):
    """
    Convenience method that wraps a :class:`TypeAdapter` around
    ``logging.getLogger(name)``

    Parameters
    ----------
    name: str
        The name of the logger to get. Passed to :func:`logging.getLogger`.
        Should probably be ``__name__``.
    """
    return TypeAdapter(logging.getLogger(name))
