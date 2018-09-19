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
                                  PassingLoggerAdapter)


class LogHandler(logging.NullHandler):
    def set_test(self, test):
        self.test = test

    def handle(self, record):
        self.test(record)


@pytest.fixture(scope='module')
def logger():
    logger_ = logging.getLogger(__name__)
    logger_.setLevel(1)
    handler = LogHandler(1)
    logger_.addHandler(handler)
    return logger_, handler


@given(name=st.text())
def test_get_logger(name):
    default_logger = logging.getLogger(name)
    vm_logger = get_logger(name)
    assert vm_logger.logger is default_logger


@example(args=[], type='general', default_type='general')
@given(args=st.lists(st.text(), min_size=0, max_size=5),
       type=st.one_of(st.none(), st.text()),
       default_type=st.text(min_size=1))
def test_type_adapter(logger, args, type, default_type):
    def test(record):
        rectype = getattr(record, 'type', None)
        if type is None:
            assert rectype == default_type
        else:
            assert rectype == type
    logger, handler = logger
    logger = TypeAdapter(logger, default_type=default_type)
    handler.set_test(test)
    fmt = ['%s']*len(args)
    fmt = ' '.join(fmt)

    note('fmt={}'.format(fmt))

    if type is None:
        logger.info(fmt, *args)
        event('type is None')
    else:
        logger.info(fmt, *args, type=type)


@example(args=[], kwargs={}, type='general', default_type='general')
@given(args=st.lists(st.text(), min_size=0, max_size=5),
       kwargs=st.dictionaries(st.text(alphabet=string.ascii_letters, min_size=1),
                              st.text(), min_size=0, max_size=5),
       type=st.one_of(st.none(), st.text()),
       default_type=st.text(min_size=1))
def test_style_type_adapter(logger, args, kwargs, type, default_type):
    def test(record):
        rectype = getattr(record, 'type', None)
        print(rectype, type, default_type)
        if type is None:
            return rectype == default_type
        else:
            return rectype == type
    logger, handler = logger
    logger = TypeAdapter(logger, default_type=default_type)
    logger = StyleAdapter(logger)
    handler.set_test(test)
    fmt = ['{}']*len(args) + ['{'+name+'}' for name in kwargs]
    fmt = ' '.join(fmt)

    note('fmt={}'.format(fmt))

    if type is None:
        logger.info(fmt, *args, **kwargs)
        event('type is None')
    else:   
        logger.info(fmt, *args, type=type, **kwargs)


@example(args=[], kwargs={})
@given(args=st.lists(st.text(), min_size=0, max_size=5),
       kwargs=st.dictionaries(st.text(alphabet=string.ascii_letters, min_size=1),
                              st.text(), min_size=0, max_size=5),
      )
def test_style_adapter(logger, args, kwargs):
    def test(record):
        assert record.getMessage() == expected
    logger, handler = logger
    logger = StyleAdapter(logger)
    handler.set_test(test)
    fmt = ['{}']*len(args) + ['{'+name+'}' for name in kwargs]
    fmt = ' '.join(fmt)

    note('fmt={}'.format(fmt))
    expected = fmt.format(*args, **kwargs)

    logger.info(fmt, *args, **kwargs)


@given(args=st.lists(st.text(), min_size=0, max_size=5),
       kwargs=st.dictionaries(st.text(alphabet=string.ascii_letters, min_size=1),
                              st.text(), min_size=1, max_size=5),
      )
def test_passing_adapter(logger, args, kwargs):
    logger, handler = logger
    handler.set_test(lambda: None)
    logger = PassingLoggerAdapter(logger)
    fmt = ['%s']*len(args) + ['%('+name+')s' for name in kwargs]
    with pytest.raises(TypeError):
        logger.info(fmt, *args, **kwargs)
