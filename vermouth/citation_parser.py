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
import re

class BibTexDirector():
    """
    Lightweight parser for BibTex files. BibTex files
    in general have an assorment of entries that
    describe the corresponding sort of publication
    to refer to and then a number required and optional
    fields for the different types of entries. A field
    for example would be Title giving the title of a
    publication. The syntax in general looks as follows:

    @<entry>{<some custom ID>, field = {<content>},
                               field = {<content>}}

    Alternatively the {} can be replaced by quotation
    marks.

    This parser only parses the version with {} as
    used by google scholar. In addition we do not
    check for missing fields or invalid fields. All
    fields are accepted and no fields are required.
    """
    def __init__(self, force_field):
        self.force_field = force_field
        self.known_entries = ["article",
                              "book",
                              "booklet",
                              "conference",
                              "inbook",
                              "incollection",
                              "inproceedings",
                              "manual",
                              "mastersthesis",
                              "misc",
                              "phdthesis",
                              "proceedings",
                              "techreport",
                              "unpublished"
                             ]

    @staticmethod
    def prepare_file(lines):
        """
        Bibtex is not sensitive to line spacing so we join
        the line as one string. Comment characters are not
        allowed.
        """
        return " ".join(line.strip() for line in lines)

    @staticmethod
    def find_entries(citation_string):
        """
        Look in a string where `@` indicates the
        beginning of a new entry and return the indices.

        Parameters
        -----------
        citation_string: str

        Yields
        --------
        int
            position of '@' in citation_string
        """
        for idx, token in enumerate(citation_string):
            if token == "@":
                yield idx

    def pop_entry_type(self, entry_string):
        """
        Given a string describing a single
        entry strip that entry from the string
        and return it. Note the string MUST
        contain the @.

        Parameters
        ------------
            entry_string: str

        Returns
        ---------
        str
            The entry type
        str
            The shortened string
        """
        assert entry_string[0] == "@"
        entry_type = entry_string[1:entry_string.find('{')]
        assert entry_type in self.known_entries
        entry_string = entry_string[len(entry_type)+1:]
        return entry_type, entry_string

    @staticmethod
    def pop_key(entry_string):
        """
        Given a string of a single entry from which the
        entry_type has already been removed (see pop_entry_type)
        get the custom ID, strip it and return the entry_string
        without that ID.

        Parameters
        -----------
        entry_string: str

        Returns
        --------
        str, str
            the key and the string without key
        """
        key_idx = entry_string.find(',')
        return entry_string[:key_idx].strip("{").strip(" "), entry_string[key_idx:]

    @staticmethod
    def extract_fields(entry_string):
        """
        Given an entry string without entry type and identified
        (i.e. ,<field_type> = {<content>}, etc.) split all the
        contents and field-types using a regular expression.

        Parameters
        -----------
        entry_string: str

        Yields
        -------
        str, str
            the field type, the field content
        """
        for field, value in re.findall("(.*?)=(.*?)\}", entry_string):
            yield field.strip(",").strip(" "), value.strip("{").strip("}")

    def parse_entry(self, entry_string):
        """
        Given a string describing a single entry, parse it and
        then update the force_field citations dict with a field
        dict.
        """
        entry_type, entry_string = self.pop_entry_type(entry_string)
        cite_key, entry_string = self.pop_key(entry_string)
        field_dict = dict(self.extract_fields(entry_string))
        field_dict["type"] = entry_type
        self.force_field.citations[cite_key] = field_dict

    def parse(self, lines):
        """
        Given lines from a bibtex file parse them and update
        the force-field citation instance variable.
        """
        # convert file to string deleting end of line charcters
        citations_string = self.prepare_file(lines)
        # extract the entries from the string
        entries = list(self.find_entries(citations_string))
        entries.append(len(citations_string))
        # parse each entry to generate a citation
        for idx, jdx in zip(entries[:-1], entries[1:]):
            self.parse_entry(citations_string[idx:jdx])
        return self.force_field.citations

def read_bib(lines, force_field):
    director = BibTexDirector(force_field=force_field)
    return director.parse(iter(lines))

def citation_formatter(citation, title=False):
    """
    Very basic and minimal formatter for citations. It
    is adopted from basic ACS style formatting. Fields within
    [] are optional.

    <authors> [journal] <year>; [doi]

    Note that the formatter cannot fromat latex
    like syntax (e.g. a{\"} for ae)
    """
    # first we split the author-list
    citation_string = ""
    # the spaces around the and are required!
    for match in citation["author"].split(" and "):
        last_name, first_names = match.split(",", 1)
        citation_string += last_name.strip() + ","
        for name in first_names.strip().split(' '):
            citation_string += " " + name.strip()[0]

        citation_string += "; "

    if "journal" in citation:
        citation_string += " " + citation["journal"]

    citation_string += " " + citation["year"]

    if "doi" in citation:
        citation_string += "; " + citation["doi"]

    return citation_string
