; Copyright 2018 University of Groningen
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;    http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

graph [
    name "BB"
    node [
        id 0
        label "N"
         ]
    node [
        id 1
        label "C"
         ]
    node [
        id 2
        label "O"
         ]
    node [
        id 3
        label "CA"
         ]
    edge [ source 0 target 3 ]
    edge [ source 1 target 2 ]
    edge [ source 1 target 3 ]
]
graph [
    name "SC1"
    node [
        id 0
        label "CB"
         ]
    node [
        id 1
        label "CG"
         ]
    node [
        id 2
        label "CD1"
         ]
    node [
        id 3
        label "CD2"
         ]
    edge [ source 0 target 1 ]
    edge [ source 1 target 2 ]
    edge [ source 1 target 3 ]
]
