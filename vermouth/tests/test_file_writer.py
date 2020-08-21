# Copyright 2020 University of Groningen
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
Contains tests for the DeferredFileWriter
"""
from pathlib import Path
import os
import pytest
from vermouth.file_writer import DeferredFileWriter


def test_is_singleton():
    """Ensure the DeferredFileWriter is a singleton"""
    writer1 = DeferredFileWriter()
    writer2 = DeferredFileWriter()

    assert writer1 is writer2


@pytest.mark.parametrize('name, existing_files, expected', [
    ('a', [], []),
    ('a', ['a'], ['#a.1#']),
    ('a.txt', ['a.txt'], ['#a.txt.1#']),
    ('a.txt', ['a'], ['a']),
    ('a.txt', ['a.txt', '#a.txt.1#'], ['#a.txt.2#', '#a.txt.1#']),
    ('a.txt', ['a.txt', '#a.txt.2#'], ['#a.txt.1#', '#a.txt.2#']),
])
def test_backup(tmpdir, monkeypatch, name, existing_files, expected):
    """
    Ensure the DeferredFileWriter backs up existing files correctly, and at the
    correct moment
    """
    monkeypatch.chdir(tmpdir)
    for idx, file in enumerate(existing_files):
        with open(file, 'w') as handle:
            handle.write(str(idx))

    writer = DeferredFileWriter()
    with writer.open(name, 'w') as handle:
        handle.write("new {}".format(name))
    writer.write()

    assert Path(name).is_file()
    with open(name) as file:
        assert file.read() == "new {}".format(name)

    for idx, name in enumerate(expected):
        assert Path(name).is_file()
        with open(name) as file:
            assert file.read() == str(idx)


def test_deferred_writing(tmpdir, monkeypatch):
    """
    Ensure the DeferredFileWriter writes changes to files at the correct moment
    """
    monkeypatch.chdir(tmpdir)

    file_name = Path('my_file.txt')
    writer = DeferredFileWriter()
    assert not file_name.exists()
    with writer.open(file_name, 'w') as file:
        file.write('hello')
    assert not file_name.exists()
    os.chdir('..')
    writer.write()
    os.chdir(str(tmpdir))
    assert file_name.exists()
    assert file_name.read_text() == 'hello'


def test_binary_writing(tmpdir, monkeypatch):
    """Ensure the DeferredFileWriter can write and append to binary files"""
    monkeypatch.chdir(tmpdir)
    file_name = Path('my_file.txt')
    writer = DeferredFileWriter()
    assert not file_name.exists()

    with writer.open(file_name, 'wb') as file:
        file.write(b'Hello')
    assert not file_name.exists()
    os.chdir('..')
    writer.write()
    os.chdir(str(tmpdir))
    assert file_name.exists()
    assert file_name.read_text() == 'Hello'

    with writer.open(file_name, 'ab') as file:
        file.write(b' world!')
    assert file_name.exists()
    assert file_name.read_text() == 'Hello'
    writer.write()
    assert file_name.read_text() == 'Hello world!'


def test_rw_plus(tmpdir, monkeypatch):
    """Ensure the DeferredFileWriter can deal with mode r+"""
    monkeypatch.chdir(tmpdir)
    path = Path('file.txt')
    path.write_text('123')
    writer = DeferredFileWriter()

    with writer.open(path, 'r+') as file:
        assert file.read() == '123'
        file.write('456')
        file.seek(0)
        assert file.read() == '123456'
    assert path.read_text() == '123'
    writer.write()

    assert path.read_text() == '123456'

@pytest.mark.parametrize('mode, exception', (
    ['o', KeyError],
))
def test_mode_errors(mode, exception):
    """
    Ensure the DeferredFileWriter raises an appropriate error for unknown file
    modes
    """
    writer = DeferredFileWriter()
    with pytest.raises(exception):
        writer.open('somefile.txt', mode)


def test_append(tmpdir, monkeypatch):
    """Ensure the DeferredFileWriter can append"""
    monkeypatch.chdir(tmpdir)
    path = Path('file.txt')
    path.write_text('123')

    writer = DeferredFileWriter()
    with writer.open(path, 'a') as file:
        file.write('abc')

    assert path.read_text() == '123'

    writer.write()

    assert path.read_text() == '123abc'


def test_closing(tmpdir, monkeypatch):
    """
    Ensure the DeferredFileWriter's close method doesn't prompt writing and
    removes any temporary files.
    """
    monkeypatch.chdir(tmpdir)
    tmpdir = Path(str(tmpdir))
    writer = DeferredFileWriter()
    monkeypatch.setattr(writer, '_tmpdir', str(tmpdir))

    assert not [p.name for p in tmpdir.iterdir()]

    with writer.open('file.txt', 'w') as file:
        file.write('abc')

    writer.write()

    assert [p.name for p in tmpdir.iterdir()] == ['file.txt']

    with writer.open('file2.txt', 'w') as file:
        file.write('abc')

    writer.close()
    assert [p.name for p in tmpdir.iterdir()] == ['file.txt']


def test_reopen(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    path = Path('file.txt')
    writer = DeferredFileWriter()

    with writer.open(path, 'w') as file:
        file.write('Hello!')
    # Close file
    with writer.open(path, 'r') as file:
        assert file.read() == 'Hello!'
