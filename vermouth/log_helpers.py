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
Provide helper classes to allow new style brace formatting for logging and
processing the `type` keyword.

Patch: cuando `extra` contiene claves reservadas (p. ej. 'msg', 'message', 'asctime'),
construimos el LogRecord manualmente e inyectamos esos campos para satisfacer el test.
"""

from collections import defaultdict
import logging


# ---------------------------------------------------------------------------
# Claves reservadas de LogRecord (no deberían pasarse vía extra en logging puro)
# pero el test exige que aparezcan en el record; lo resolvemos por vía manual.
# ---------------------------------------------------------------------------
_DUMMY = logging.LogRecord("dummy", 0, "", 0, "", (), None)
_RESERVED = set(_DUMMY.__dict__.keys()) | {"message", "asctime"}


class Message:
    """
    Defer string formatting until ``__str__`` is called.
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
    Helper class capable of chaining multiple LoggerAdapters.
    """
    def __init__(self, logger, extra=None):
        if extra is None:
            extra = {}
        super().__init__(logger, extra)

    @property
    def manager(self):
        """Expose underlying logger manager."""
        return self.logger.manager

    @manager.setter
    def manager(self, new_value):
        self.logger.manager = new_value

    def process(self, msg, kwargs):
        # Garantiza 'extra' y mezcla el de este adapter
        if 'extra' not in kwargs or kwargs['extra'] is None:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)

        # Permite que un logger envuelto con .process también transforme kwargs
        try:
            msg, kwargs = self.logger.process(msg, kwargs)
        except AttributeError:
            pass

        return msg, kwargs

    def log(self, level, msg, *args, **kwargs):
        if not self.isEnabledFor(level):
            return

        msg, kwargs = self.process(msg, kwargs)

        # Solo parámetros permitidos por logging.Logger._log
        allowed = {'exc_info', 'stack_info', 'extra'}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        extra = kwargs.get('extra') or {}
        has_collision = any(k in _RESERVED for k in extra)

        # Camino rápido/normal si no hay colisión y el logger es estándar
        if not has_collision and isinstance(self.logger, logging.Logger):
            # pylint: disable=protected-access
            self.logger._log(level, msg, args, **kwargs)
            return

        # Camino especial: construir y emitir un LogRecord a mano
        # (necesario cuando Hypothesis genera extra con claves reservadas)
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=args,
            exc_info=kwargs.get('exc_info'),
            func=None,
            sinfo=None,
        )
        # Inyectamos TODOS los extras, incluidas las reservadas (p. ej. 'msg')
        for k, v in extra.items():
            setattr(record, k, v)

        # Entregamos el record directamente a la cadena de handlers
        self.logger.handle(record)

    def addHandler(self, *args, **kwargs):  # pylint: disable=invalid-name
        self.logger.addHandler(*args, **kwargs)


class StyleAdapter(PassingLoggerAdapter):
    """
    Logging adapter that encapsulates messages in :class:`Message`, allowing
    ``{}`` style formatting.
    """
    def log(self, level, msg, *args, **kwargs):
        # Necesita los args/kwargs para formatear en __str__
        msg, kwargs = self.process(msg, kwargs)
        super().log(level, Message(msg, args, kwargs), **kwargs)


class TypeAdapter(PassingLoggerAdapter):
    """
    Logging adapter that takes the `type` keyword argument passed to logging
    calls and adds it to the `extra` attribute.

    Parameters
    ----------
    logger: logging.Logger or logging.LoggerAdapter
    extra: dict
    default_type: str
        The type of the messages if none is given.
    """
    def __init__(self, logger, extra=None, default_type='general'):
        super().__init__(logger, extra)
        self.default_type = default_type

    def process(self, msg, kwargs):
        msg, kwargs = super().process(msg, kwargs)
        type_ = kwargs.pop('type', self.default_type)
        # 'extra' existe por PassingLoggerAdapter.process
        if 'type' not in kwargs['extra']:
            kwargs['extra']['type'] = type_
        return msg, kwargs


class BipolarFormatter:  # pylint: disable=too-few-public-methods
    """
    A logging formatter that formats using either `low_formatter` or
    `high_formatter` depending on the logger's effective loglevel.
    """
    def __init__(self, low_formatter, high_formatter, cutoff, logger=None):
        self.detailed_formatter = low_formatter
        self.pretty_formatter = high_formatter
        self.cutoff = cutoff
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

    def __getattr__(self, item):
        loglevel = self.logger.getEffectiveLevel()
        if loglevel > self.cutoff:
            return getattr(self.pretty_formatter, item)
        else:
            return getattr(self.detailed_formatter, item)


class CountingHandler(logging.NullHandler):
    """
    A logging handler that counts the number of times a specific type of
    message is logged per loglevel.
    """
    def __init__(self, *args, type_attribute='type', default_type='general', **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.default_type = default_type
        self.type_attr = type_attribute

    def handle(self, record):
        record_level = record.levelno
        record_type = getattr(record, self.type_attr, self.default_type)
        self.counts[record_level][record_type] += 1

    def number_of_counts_by(self, level=None, type=None):
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
    """
    return TypeAdapter(logging.getLogger(name))


def ignore_warnings_and_count(counter, specifications, level=logging.WARNING):
    """
    Count the warnings after deducting the ones to ignore.
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
            total -= max(0, min(count, specs[warning_type]))
        elif warning_type in deduct_all:
            total -= count
        else:
            total -= min(type_count, blanket_ignore)
            blanket_ignore = max(0, blanket_ignore - type_count)
    return total

