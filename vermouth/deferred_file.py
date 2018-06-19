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

import os
import sys
import tempfile

# TODO: replace with pathlib.replace, and add pathlib2 to requirements for py2.7.
# from https://stupidpythonideas.blogspot.nl/2014/07/getting-atomic-writes-right.html
try:
    replace = os.replace  # pylint: disable=invalid-name
except AttributeError:
    if sys.platform == 'win32':
        import win32api  # pylint: disable=import-error
        import win32con  # pylint: disable=import-error

        def replace(src, dst):
            win32api.MoveFileEx(src, dst, win32con.MOVEFILE_REPLACE_EXISTING)
    else:
        replace = os.rename  # pylint: disable=invalid-name


def find_next_filename(basename, suffix='bak'):
    def count_to_ext(count):
        if count < 0:
            return ''
        elif count == 0:
            return os.path.extsep + suffix
        else:
            return '{extsep}{suffix}{extsep}{n}'.format(extsep=os.path.extsep,
                                                        suffix=suffix, n=count)
    count = -1
    filename = basename
    while os.path.exists(filename):
        count += 1
        filename = basename + count_to_ext(count)
    return filename


def replace_with_backup(file_in, file_out, suffix='bak'):
    if os.path.exists(file_out):
        file_out_bak = find_next_filename(file_out, suffix=suffix)
        print('Backing up {} to {}'.format(file_out, file_out_bak))
        replace(file_out, file_out_bak)
    return replace(file_in, file_out)


class DeferredFile:
    def __init__(self, filename):
        self.filename = filename
        self.approved = False
        self._tempfile = None

    def open(self):
        self._tempfile = tempfile.NamedTemporaryFile(mode='w', delete=False)

    def approve(self):
        self.approved = True

    def close(self):
        self._tempfile.close()
        if self.approved:
            replace_with_backup(self._tempfile.name, self.filename)
        else:
            os.remove(self._tempfile.name)

    # For some reason tempfile.NamedTemporaryFile is a function, not a class.
    # And to top it off, the object it returns has a magic getattr method...
    def write(self, *args, **kwargs):
        self._tempfile.write(*args, **kwargs)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:  # An exception happened
            self.approved = False
        self.close()
        return False  # re-raise any exceptions


# if __name__ == '__main__':
#     dfile = DeferredFile('flup1.txt')
#     dfile.open()
#     dfile.write('boe')
#     dfile.close()
#
#     with DeferredFile('flup.txt') as df:
#         df.write('Hello!')
#
#     with DeferredFile('flup2.txt') as df:
#         df.write('World!')
#         df.approve()
