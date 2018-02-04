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

import itertools
from glob import glob
import os
from .gmx.rtp import read_rtp
from .ffinput import read_ff
from . import DATA_PATH


class ForceField(object):
    def __init__(self, directory):
        parsers = {'.rtp': read_rtp, '.ff': read_ff}
        source_files = itertools.chain(
            glob(os.path.join(directory, '*.ff')),
            glob(os.path.join(directory, '*.rtp')),
        )
        blocks = {}
        links = []
        for source in source_files:
            extension = os.path.splitext(source)[-1]
            with open(source) as infile:
                file_blocks, file_links = parsers[extension](infile)
            blocks.update(file_blocks)
            links.extend(file_links)

        self.name = os.path.basename(directory)
        self.blocks = blocks
        self.links = links
        self.reference_graphs = blocks


def find_force_fields(directory):
    """
    Find all the force fields in the given directory.

    A force field is defined as a directory that contains at least one RTP
    file. The name of the force field is the base name of the directory.
    """
    force_fields = {}
    directory = str(directory)  # Py<3.6 compliance
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if glob(os.path.join(path, '*.rtp')) + glob(os.path.join(path, '*.ff')):
                force_fields[name] = ForceField(path)
    return force_fields


FORCE_FIELDS = find_force_fields(os.path.join(DATA_PATH, 'force_fields'))
