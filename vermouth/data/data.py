# Copyright 2025 University of Groningen
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

from .. import DATA_PATH
from ..citation_parser import read_bib


def _read_quote_file(filehandle):
    """
    Iterates over `filehandle`, and yields all strings that are not empty.

    Parameters
    ----------
    filehandle: collections.abc.Iterable[str]
        A file opened for reading.

    Yields
    ------
    str
        All stripped elements of `filehandle` that are not empty.
    """
    for line in filehandle:
        line = line.strip()
        if line:
            yield line


with open(DATA_PATH/'citations.bib') as citation_file:
    COMMON_CITATIONS = read_bib(citation_file)

with open(DATA_PATH/'quotes.txt') as quotes_file:
    QUOTES = list(_read_quote_file(quotes_file))
