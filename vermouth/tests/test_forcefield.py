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
Tests for the ForceField class.
"""

import pathlib
import pytest
import vermouth.forcefield
import vermouth.molecule

# pylint: disable=redefined-outer-name

@pytest.fixture
def empty_force_field():
    """
    Build an empty force field named 'empty'.
    """
    return vermouth.forcefield.ForceField(name='empty')


@pytest.fixture
def force_field_with_features(empty_force_field):
    """
    Build a force field with links describing features.

    The force field is empty except for the links. The links are empty except
    for the feature lists.
    """
    features = (
        ['single_0'], ['single_1'], ['single_2'],
        ['single_0', 'multi_0', 'multi_1'],
        ['multi_0', 'multi_2'],
        [],
    )
    for feature_names in features:
        link = vermouth.molecule.Link()
        link.features = feature_names
        empty_force_field.links.append(link)
    return empty_force_field


@pytest.mark.parametrize('attribute, value', (
    ('name', 'empty'),
    ('blocks', {}),
    ('links', []),
    ('modifications', {}),
    ('renamed_residues', {}),
    ('variables', {}),
    ('reference_graphs', {}),
    ('features', set()),
))
def test_init_empty_force_field(empty_force_field, attribute, value):
    """
    Test that an empty force field has the expected attributes.
    """
    assert getattr(empty_force_field, attribute) == value


def test_features(force_field_with_features):
    """
    Test that link features are found by the force field.
    """
    expected = {'single_0', 'single_1', 'single_2',
                'multi_0', 'multi_1', 'multi_2'}
    assert force_field_with_features.features == expected


@pytest.mark.parametrize('name', (
    'single_0', 'single_1', 'single_2', 'multi_0', 'multi_1', 'multi_2'
))
def test_has_feature_true(force_field_with_features, name):
    """
    Test that :attr:`vermouth.forcefield.ForceField.has_feature` finds the
    expected features.
    """
    assert force_field_with_features.has_feature(name)


def test_has_feature_false(force_field_with_features):
    """
    Test that :attr:`vermouth.forcefield.ForceField.has_feature` does not find
    features that should not be there.
    """
    assert not force_field_with_features.has_feature('absent')


def test_create_ff_from_dir_name(tmpdir):
    """
    Creates a force field from a directory AND a name, assure that the correct
    name is used.

    The name provided with the 'name' argument should be used.
    """
    directory = tmpdir.mkdir('dirname')
    name = 'the_name'
    ff = vermouth.forcefield.ForceField(
        directory=str(directory),
        name=name,
    )
    assert ff.name == name


def test_create_ff_from_name():
    """
    Creates a force field by name (no directory) and make sure it has the
    expected name.
    """
    name = 'the_ff_name'
    ff = vermouth.forcefield.ForceField(name=name)
    assert ff.name == name


@pytest.mark.parametrize('path_type', (str, pathlib.Path))
def test_create_ff_from_dir(tmpdir, path_type):
    """
    Creates a force field from a directory, assure that the name is correct.
    """
    ff_name = 'name'
    # Depending on the version of python, the type of tmpdir may differ.
    directory = path_type(str(tmpdir.mkdir(ff_name)))
    ff = vermouth.forcefield.ForceField(directory=directory)
    assert ff.name == ff_name


def test_init_empty_force_field_error_neither():
    """
    Test that an error is raised when neither directory nor name are given to
    ForceField.
    """
    with pytest.raises(TypeError):
        vermouth.forcefield.ForceField()
