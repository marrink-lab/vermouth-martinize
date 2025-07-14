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

from vermouth.parser_utils import SectionParser

# pylint: disable=missing-function-docstring, missing-class-docstring, no-member, invalid-name

def test_metaclass_inheritance():
    """
    Test that the (populated) METH_DICT from the SectionParser metaclass is
    correctly inherited or reset by subclasses
    """
    class A(metaclass=SectionParser):
        def method_a(self):
            pass

        @SectionParser.section_parser('section_a')
        def method_a2(self):
            pass

    class B(A):
        @SectionParser.section_parser('section_b')
        def method_b1(self):
            pass

    class C(A):
        @SectionParser.section_parser('section_c')
        def method_c1(self):
            pass

    assert A.METH_DICT == {('section_a', ): (A.method_a2, {})}
    assert B.METH_DICT == {('section_a',): (A.method_a2, {}), ('section_b',): (B.method_b1, {})}
    assert C.METH_DICT == {('section_a',): (A.method_a2, {}), ('section_c',): (C.method_c1, {})}

    class C2(C):
        @SectionParser.section_parser('section_c')
        def method_c2(self):
            pass

    assert C.METH_DICT == {('section_a',): (A.method_a2, {}), ('section_c',): (C.method_c1, {})}
    assert C2.METH_DICT == {('section_a',): (A.method_a2, {}), ('section_c',): (C2.method_c2, {})}
