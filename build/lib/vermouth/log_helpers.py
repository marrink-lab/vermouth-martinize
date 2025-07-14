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
from collections import defaultdict
import logging


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


class PassingLoggerAdapter(logging.LoggerAdapter):
    """
    Helper class that is actually capable of chaining multiple LoggerAdapters.
    """
    def __init__(self, logger, extra=None):
        if extra is None:
            extra = {}
        # These are all set by the logger.setter property. Which
        # super().__init__ calls.
        super().__init__(logger, extra)
        # A LoggerAdapter does not have a manager, but logging.Logger.log needs
        # it to see if logging is enabled.

    @property
    def manager(self):
        """
        .. autoattribute:: logging.Logger.manager
        """
        return self.logger.manager

    @manager.setter
    def manager(self, new_value):
        self.logger.manager = new_value

    def process(self, msg, kwargs):
        # The documentation is a lie and the original implementation clobbers
        # 'extra' that is set by other LoggerAdapters in the chain.
        # LoggerAdapter's process method is FUBARed, and aliases kwargs and
        # self.extra. And that's all it does. So we do it here by hand to make
        # sure we actually have an 'extra' attribute.
        # It should maybe be noted that generally this method gets executed
        # multiple times, so occasionally self.extra items are very persistent.
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        try:
            # logging.Logger does not have a process.
            msg, kwargs = self.logger.process(msg, kwargs)
        except AttributeError:
            pass
        return msg, kwargs

    def log(self, level, msg, *args, **kwargs):
        # Differs from super().log because this calls `self.logger.log` instead
        # of self.logger._log. LoggerAdapters don't have a _log.
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            if isinstance(self.logger, logging.Logger):
                # logging.Logger.log throws a hissy fit if it gets too many
                # kwargs, so leave just the ones known.
                kwargs = {key: val for key, val in kwargs.items()
                          if key in ['level', 'msg', 'exc_info', 'stack_info', 'extra']}
                self.logger._log(level, msg, args, **kwargs)  # pylint: disable=protected-access
            else:
                self.logger.log(level, msg, *args, **kwargs)

    def addHandler(self, *args, **kwargs):  # pylint: disable=invalid-name
        self.logger.addHandler(*args, **kwargs)


class StyleAdapter(PassingLoggerAdapter):
    """
    Logging adapter that encapsulate messages in :class:`Message`, allowing
    ``{}`` style formatting.
    """
    def log(self, level, msg, *args, **kwargs):
        # We need a different `log` method, since `Message` needs the args
        # as well as the kwargs. Otherwise it could've been done in process.
        # You can probably work around that by giving Message a __mod__ method,
        # but that's too much effort for now.
        msg, kwargs = self.process(msg, kwargs)
        super().log(level, Message(msg, args, kwargs), **kwargs)


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
        super().__init__(logger, extra)
        self.default_type = default_type

    def process(self, msg, kwargs):
        msg, kwargs = super().process(msg, kwargs)
        type_ = kwargs.pop('type', self.default_type)
        if 'type' not in kwargs['extra']:
            kwargs['extra']['type'] = type_
        return msg, kwargs


class BipolarFormatter:  # pylint: disable=too-few-public-methods
    """
    A logging formatter that formats using either `low_formatter` or
    `high_formatter` depending on the `logger`'s effective loglevel.

    Parameters
    ----------
    low_formatter: logging.Formatter
        The formatter used if `cutoff` <= `logger.getEffectiveLevel()`.
    high_formatter: logging.Formatter
        The formatter used if `cutoff` > `logger.getEffectiveLevel()`.
    cutoff: int
        The cutoff used to decide whether the low or high formatter is used.
    logger: logging.Logger
        The logger whose effective loglevel is used. Defaults to
        ``logging.getLogger()``.
    """
    def __init__(self, low_formatter, high_formatter, cutoff, logger=None):
        self.detailed_formatter = low_formatter
        self.pretty_formatter = high_formatter
        self.cutoff = cutoff
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

    def __getattr__(self, item):
        # Delegate to pretty formatter if the loglevel is high enough, and the
        # detailed formatter otherwise
        loglevel = self.logger.getEffectiveLevel()
        if loglevel > self.cutoff:
            return getattr(self.pretty_formatter, item)
        else:
            return getattr(self.detailed_formatter, item)


class CountingHandler(logging.NullHandler):
    """
    A logging handler that counts the number of times a specific type of
    message is logged per loglevel.

    Parameters
    ----------
    type_attribute: str
        The name of the attribute carrying the type.
    default_type: str
        The type of message if none is provided.

    """
    def __init__(self, *args, type_attribute='type', default_type='general', **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.default_type = default_type
        self.type_attr = type_attribute

    def handle(self, record):
        """
        Handle a log record by counting it.
        """
        record_level = record.levelno
        record_type = getattr(record, self.type_attr, self.default_type)
        self.counts[record_level][record_type] += 1

    def number_of_counts_by(self, level=None, type=None):
        """
        Return the number of logging calls counted, filtered by level and type.

        Parameters
        ----------
        level
            Only count log events of this level.
        type
            Only count log events of this type.

        Returns
        -------
        int
            The number of events counted.
        """
        out = 0
        for lvl, type_counts in self.counts.items():
            if level is not None and lvl < level:
                continue
            for type_, count in type_counts.items():
                if type is not None and type != type_:
                    continue
                out += count
        return out


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


def ignore_warnings_and_count(counter, specifications, level=logging.WARNING):
    """
    Count the warnings after deducting the ones to ignore.

    Warnings to ignore are specified as tuple ``(<warning-type>, <count>)``.
    The count is ``None`` if all warnings of that type should be ignored,
    and the warning type is ``None`` to indicate that the count is about
    all not specified types.

    In case the same type is specified more than once, only the higher
    count is used.
    """
    number_of_warnings = counter.number_of_counts_by(level=level)
    specs = {}
    deduct_all = set()
    for spec_parts in specifications:
        for warning_type, count in spec_parts:
            if count is None:
                deduct_all.add(warning_type)
            else:
                specs[warning_type] = max(specs.get(warning_type, 0), count)
    blanket_ignore = specs.get(None, 0)
    warning_count = counter.counts[level]
    total = number_of_warnings
    for warning_type, count in warning_count.items():
        type_count = warning_count[warning_type]
        if warning_type in specs:
            # Subtract at least 0, and at most the number of warnings counted so
            # the resulting total is guaranteed to be between 0 and `count`.
            total -= max(0, min(count, specs[warning_type]))
        elif warning_type in deduct_all:
            total -= count
        else:
            total -= min(type_count, blanket_ignore)
            blanket_ignore = max(0, blanket_ignore - type_count)
    return total
