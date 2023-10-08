# Copyright 2023 University of Groningen
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
Read RCSU Go model contact maps.
"""

def read_go_map(file_path, header_lines=0, cols=[2, 6, 10]):
    """
    Read a RCSU contact map.
    """
    with open(file_path, "r", encoding='UTF-8') as _file:
        lines = _file.readlines()

    contacts = []
    read = False
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) == 0:
            continue
        # we start parsing
        # using R1 is super flaky but I don't see a good way to identify
        # when the contact map starts ...
        if tokens[0] == "R1":
            read = True
        elif read:
        # a contact consits of a resid and the chain ID
            contacts.append((int(tokens[1]), tokens[2], int(tokens[4]), tokens[5]))
        else:
            continue
    return contacts
