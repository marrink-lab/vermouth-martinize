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
import inspect
import logging


class PassingLoggerAdapter(logging.LoggerAdapter):
    """
    Helper class that figures out which keyword arguments are known by the log
    method of this :class:`logging.LoggerAdapter`, and it's `logger` (which
    might be another :class:`logging.LoggerAdapter`). Subclasses can use this
    information to remove unknown keywords from their `super().log` calls.

    Attributes
    ----------
    known_kwargs: set[str]
        A union of all keywords known by this object's `log` method and
        :attr:`child_kwargs`.
    child_kwargs: set[str]
        A set of all keywords known by this object's `logger`'s `log` method.
    """
    def __init__(self, logger, extra=None):
        if extra is None:
            extra = {}
        super().__init__(logger, extra)

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, new_val):
        self._logger = new_val
        self._find_kwargs()

    def _find_kwargs(self):
        logsig = inspect.signature(self.log)
        my_kwargs = set(name for name, param in logsig.parameters.items()
                        if param.kind not in [param.VAR_POSITIONAL, param.VAR_KEYWORD])
        if isinstance(self._logger, logging.Logger):
            child_kwargs = {'lvl', 'msg', 'exc_info', 'stack_info', 'extra'}
        else:
            if hasattr(self._logger, 'known_kwargs'):
                child_kwargs = self._logger.known_kwargs
            else:
                parameters = inspect.signature(self._logger.log).parameters
                child_kwargs = set(name for name, param in parameters.items()
                                   if param.kind not in [param.VAR_POSITIONAL, param.VAR_KEYWORD])
        self.known_kwargs = my_kwargs.union(child_kwargs)
        self.child_kwargs = child_kwargs


class Message:
    """
    Class that defers string formatting until it's ``__str__`` method is
    called.
    """
    def __init__(self, fmt, args, kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)

    def __repr__(self):
        return '"{}".format(*{}, **{})'.format(self.fmt, self.args, self.kwargs)


class StyleAdapter(PassingLoggerAdapter):
    """
    Logging adapter that encapsulate messages in :class:`Message`, allowing
    ``{}`` style formatting.
    """
    def log(self, lvl, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        chain_kwargs = {key: val for key, val in kwargs.items()
                        if key in self.child_kwargs}
        self.logger.log(lvl, Message(msg, args, kwargs), **chain_kwargs)


class TypeAdapter(PassingLoggerAdapter):
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
        self.default_type = default_type
        super().__init__(logger, extra)

    def log(self, *args, type=None, **kwargs):  # pylint: disable=arguments-differ
        if type is None:
            type = self.default_type
        if 'extra' not in kwargs:
            kwargs['extra'] = dict(type=type)
        elif 'type' not in kwargs['extra']:
            kwargs['extra']['type'] = type
        self.logger.log(*args, **kwargs)


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
