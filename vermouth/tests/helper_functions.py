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
Contains helper functions for tests.
"""


def make_into_set(iter_of_dict):
    """
    Convenience function that turns an iterator of dicts into a set of
    frozenset of the dict items.
    """
    return set(frozenset(dict_.items()) for dict_ in iter_of_dict)
