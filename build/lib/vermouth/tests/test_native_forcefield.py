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
Tests for vermouth.forcefield.get_native_force_field.

These tests cannot be in the same module as the rest of the tests for the
vermouth.forcefield module. Indeed, the tests in this file causes to load all
force fields which create test coverage noise.
"""

import pytest
import vermouth.forcefield


def test_get_native_force_field_identity():
    """
    Test that the same instance is returned for multiple request of the same
    force field.
    """
    # We could parametrize the test to check each force field separatly, but we
    # need to make sure that different requested name result in different
    # instances. Without that, the test would pass if the function was always
    # returning the same object regardless of the argument.
    requests = ('charmm', 'martini22', 'martini22p')
    first_call = set(id(vermouth.forcefield.get_native_force_field(name))
                     for name in requests)
    second_call = set(id(vermouth.forcefield.get_native_force_field(name))
                      for name in requests)

    # Are all the force fields different?
    assert len(first_call) == len(second_call) == len(requests)
    # Are the same instances returned for the two calls?
    assert first_call == second_call


def test_get_native_force_field_key_error():
    """
    Test that get_native_force_field raises a KeyError for unfound force fields.
    """
    with pytest.raises(KeyError):
        vermouth.forcefield.get_native_force_field('non existant')
