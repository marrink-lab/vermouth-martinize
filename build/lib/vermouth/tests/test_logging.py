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
Tests for the log helpers
"""

import logging
import string

from hypothesis import strategies as st
from hypothesis import given, note, event, example
import pytest

from vermouth.log_helpers import (get_logger, TypeAdapter, StyleAdapter,
                                  PassingLoggerAdapter, Message,
                                  BipolarFormatter, CountingHandler,
                                  ignore_warnings_and_count,)

# Pylint does *not* like pytest fixtures... Also, sometimes you just need more
# data
# pylint: disable=redefined-outer-name, too-many-arguments

KWARG_ST = st.text(alphabet=string.ascii_letters, min_size=1)


class FormatCounter(str):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._format_count = 0

    def format(self, *args, **kwargs):
        """"Increments the counter, and format self."""
        self._format_count += 1
        return self.__class__(super().format(*args, **kwargs))

    def __mod__(self, args):
        self._format_count += 1
        return self.__class__(super().__mod__(args))

    def get_count(self):
        """Returns the current value of the counter"""
        return self._format_count


class LogHandler(logging.NullHandler):
    """Helper class which will run a test for every log record"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test = lambda: None

    def set_test(self, test):
        """Set the test to be run"""
        self.test = test

    def handle(self, record):
        """Does nothing, except run the test."""
        self.test(record)


@pytest.fixture(scope='module')
def handler():
    """Sets up a LogHandler"""
    handler_ = LogHandler(1)
    return handler_


@given(name=st.text())
def test_get_logger(name):
    """
    Make sure get_logger gives the correct logger.
    """
    default_logger = logging.getLogger(name)
    vm_logger = get_logger(name)
    assert vm_logger.logger is default_logger


@example(args=[], type_='general', default_type='general', extra=None)
@given(
    args=st.lists(st.text(), min_size=0, max_size=5),
    type_=st.one_of(st.none(), st.text()),
    default_type=st.text(min_size=1),
    extra=st.one_of(
        st.none(),
        st.dictionaries(KWARG_ST, st.integers(), min_size=0, max_size=5)
    )
)
def test_type_adapter(handler, args, type_, default_type, extra):
    """Make sure the TypeAdapter sets the correct type attr"""
    def test(record):
        """Make sure the type attribute is as expected"""
        rectype = getattr(record, 'type', None)
        if rectype is None:
            assert False, "No type for record!"
        if type_ is None:
            assert rectype == default_type
        else:
            assert rectype == type_
        if extra is not None:
            note(str(dir(record)))
            for key, val in extra.items():
                assert getattr(record, key, None) == val

    logger = logging.getLogger('test_type_adapter')
    logger.setLevel(1)
    logger.addHandler(handler)
    logger = TypeAdapter(logger, default_type=default_type, extra=extra)
    handler.set_test(test)
    fmt = ['%s'] * len(args)
    fmt = ' '.join(fmt)

    note('fmt={}'.format(fmt))

    if type_ is None:
        logger.info(fmt, *args)
        event('type is None')
    else:
        logger.info(fmt, *args, type=type_)


@example(args=[], kwargs={}, type_='general', default_type='general', extra=None)
@given(
    args=st.lists(st.text(), min_size=0, max_size=5),
    kwargs=st.dictionaries(KWARG_ST,
                           st.text(), min_size=0, max_size=5),
    type_=st.one_of(st.none(), st.text()),
    default_type=st.text(min_size=1),
    extra=st.one_of(
        st.none(),
        st.dictionaries(KWARG_ST, st.integers(), min_size=0, max_size=5)
    )
)
def test_style_type_adapter(handler, args, kwargs, type_, default_type, extra):
    """Make sure that if you have both a TypeAdapter and a StyleAdapter the
    type you provide ends up in the right place, and that it doesn't interfere
    with keyword formatting."""
    def test(record):
        """Make sure the type attribute is as expected"""
        rectype = getattr(record, 'type', None)
        if rectype is None:
            assert False, "No type for record!"
        if type_ is None:
            assert rectype == default_type
        else:
            assert rectype == type_
        if extra is not None:
            for key, val in extra.items():
                assert getattr(record, key, None) == val

    logger = logging.getLogger('test_style_type_adapter')
    logger.setLevel(1)
    logger.addHandler(handler)
    logger = TypeAdapter(logger, default_type=default_type)
    logger = StyleAdapter(logger, extra=extra)
    handler.set_test(test)
    fmt = ['{}']*len(args) + ['{' + name + '}' for name in kwargs]
    fmt = ' '.join(fmt)
    print('-' * 60)

    note('fmt={}'.format(fmt))
    if type_ is None:
        logger.info(fmt, *args, **kwargs)
        event('type is None')
    else:
        logger.info(fmt, *args, type=type_, **kwargs)


@example(args=[], kwargs={}, extra=None)
@given(
    args=st.lists(st.text(), min_size=0, max_size=5),
    kwargs=st.dictionaries(KWARG_ST,
                           st.text(), min_size=0, max_size=5),
    extra=st.one_of(st.none(),
                    st.dictionaries(KWARG_ST, st.integers(), min_size=0, max_size=5))
)
def test_style_adapter(handler, args, kwargs, extra):
    """Make sure the StyleAdapter can do keyword formatting"""
    def test(record):
        """Make sure the formatting worked"""
        assert record.getMessage() == expected
        if extra is not None:
            for key, val in extra.items():
                assert getattr(record, key, None) == val

    logger = logging.getLogger('test_style_adapter')
    logger.setLevel(1)
    logger.addHandler(handler)
    logger = StyleAdapter(logger, extra=extra)
    handler.set_test(test)
    fmt = ['{}']*len(args) + ['{' + name + '}' for name in kwargs]
    fmt = ' '.join(fmt)

    note('fmt={}'.format(fmt))
    expected = fmt.format(*args, **kwargs)
    logger.info(fmt, *args, **kwargs)


@given(
    args=st.lists(st.text(), min_size=0, max_size=5),
    kwargs=st.dictionaries(KWARG_ST, st.text(), min_size=1, max_size=5),
)
def test_passing_adapter(handler, args, kwargs):
    """Make sure the PassingLoggerAdapter does not allow keywords to be set for
    formatting."""
    def test(_):
        assert False
    handler.set_test(test)
    logger = logging.getLogger('test_passing_adapter')
    logger.setLevel(1)
    logger.addHandler(handler)
    logger = PassingLoggerAdapter(logger)
    fmt = ['%s']*len(args) + ['%(' + name + ')s' for name in kwargs]
    fmt = ' '.join(fmt)
    note(fmt)
    logger.setLevel(logging.INFO)
    logger.debug('')


@given(
    args=st.lists(st.text(), min_size=0, max_size=5),
    kwargs=st.dictionaries(KWARG_ST, st.text(), min_size=0, max_size=5)
)
def test_message(args, kwargs):
    """Make sure Message doesn't formats it's contents needlessly"""
    fmt = ['{}']*len(args) + ['{' + name + '}' for name in kwargs]
    fmt = ' '.join(fmt)
    note('fmt={}'.format(fmt))
    counter = FormatCounter(fmt)
    msg = Message(counter, args, kwargs)
    assert counter.get_count() == 0
    assert str(msg) == fmt.format(*args, **kwargs)
    assert counter.get_count() == 1

    assert repr(msg) == '"{}".format(*{}, **{})'.format(fmt, args, kwargs)


def test_bipolar_formatter():
    """
    Make sure the bipolar formatter calls the correct formatter.
    """
    low_counter = FormatCounter('%(message)s')
    low_formatter = logging.Formatter(fmt=low_counter)
    high_counter = FormatCounter('%(message)s')
    high_formatter = logging.Formatter(fmt=high_counter)
    logger = logging.getLogger('test_bipolar_formatter')
    formatter = BipolarFormatter(low_formatter, high_formatter,
                                 logging.INFO, logger=logger)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    assert logger is formatter.logger

    logger.setLevel(1)

    logger.info('boo')
    logger.error('baa')
    # Logger level <= cutoff, so both should go to low_counter
    assert low_counter.get_count() == 2
    assert high_counter.get_count() == 0

    logger.setLevel(logging.WARNING)
    logger.info('boo')
    logger.warning('baa')
    # Logger level > cutoff, so both should go to high_counter, but one of the
    # messages is too low priority to be shown.
    assert low_counter.get_count() == 2
    assert high_counter.get_count() == 1


def test_bipolar_formatter_logger():
    """
    Make sure the bipolar logger picks the correct logger if none is given.
    """
    low_counter = FormatCounter('%(message)s')
    low_formatter = logging.Formatter(fmt=low_counter)
    high_counter = FormatCounter('%(message)s')
    high_formatter = logging.Formatter(fmt=high_counter)
    formatter = BipolarFormatter(low_formatter, high_formatter,
                                 logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = formatter.logger
    logger.addHandler(handler)

    expected_logger = logging.getLogger()
    assert expected_logger is logger


@pytest.mark.parametrize('level, type_, expected', (
    [None, None, 7],
    [logging.DEBUG, None, 7],
    [None, 'a', 2],
    [logging.INFO, 'general', 2],
    [None, 'd', 0]
))
def test_counter(level, type_, expected):
    """Make sure the CountingHandler has learned how to count"""
    logger = logging.getLogger('test_counter')
    handler = CountingHandler(default_type='0')
    logger.addHandler(handler)
    logger.setLevel(1)
    logger = TypeAdapter(logger)

    logger.info('')
    logger.info('')
    logger.info('', type='c')
    logger.warning('', type='a')
    logger.error('', type='a')
    logger.debug('', type='b')
    logger.debug('', type='b')
    expected_total = {logging.DEBUG: {'b': 2},
                      logging.INFO: {'general': 2, 'c': 1},
                      logging.WARNING: {'a': 1},
                      logging.ERROR: {'a': 1}}
    assert handler.counts == expected_total

    assert handler.number_of_counts_by(level=level, type=type_) == expected


@pytest.fixture
def mock_counter():
    class MockCountingHandler:
        def __init__(self):
            self.counts = {
                logging.WARNING: {
                    'default': 4,
                    'something': 8,
                    'other thing': 2,
                    'empty': 0,
                },
                logging.INFO: {
                    'something': 5,
                },
            }

        def number_of_counts_by(self, level=None, type=None):
            assert type is None, 'The mock does not discriminate by type'
            results = {
                logging.WARNING: 14,
                logging.INFO: 5,
            }
            return results.get(level, 0)

    return MockCountingHandler()


@pytest.mark.parametrize('specification, level, expected', (
    # -maxwarn 4
    ([[(None, 4)]], logging.WARNING, 10),
    # -maxwarn something
    ([[('something', None)]], logging.WARNING, 6),
    # -maxwarn something:5
    ([[('something', 5)]], logging.WARNING, 9),
    # -maxwarn something:5 2
    ([[('something', 5), (None, 2)]], logging.WARNING, 7),
    # -maxprint something:5 -maxprint 2
    ([[('something', 5)], [(None, 2)]], logging.WARNING, 7),
    # Not accessible from the command line
    ([[('something', 3)]], logging.INFO, 2),
))
def test_ignore_warnings_and_count(mock_counter, specification, level, expected):
    remaining = ignore_warnings_and_count(mock_counter, specification, level)
    assert remaining == expected


