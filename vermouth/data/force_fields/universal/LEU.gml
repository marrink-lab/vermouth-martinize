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
    name "GLY"
    node [
        id 0
        atomname "C"
    ]
    node [
        id 1
        atomname "N"
    ]
    node [
        id 2
        atomname "O"
    ]
    node [
        id 3
        atomname "CA"
    ]
    node [
        id 4
        atomname "CB"
    ]
    node [
        id 5
        atomname "CG"
    ]
    node [
        id 6
        atomname "CD1"
    ]
    node [
        id 7
        atomname "CD2"
    ]
    node [
        id 8
        atomname "HA"
    ]
    node [
        id 9
        atomname "HB1"
    ]
    node [
        id 10
        atomname "HB2"
    ]
    node [
        id 11
        atomname "HG"
    ]
    node [
        id 12
        atomname "HD11"
    ]
    node [
        id 13
        atomname "HD12"
    ]
    node [
        id 14
        atomname "HD13"
    ]
    node [
        id 15
        atomname "HD21"
    ]
    node [
        id 16
        atomname "HD22"
    ]
    node [
        id 17
        atomname "HD23"
    ]
    node [
        id 18
        atomname "HN"
    ]
edge [ source 0 target 3]
edge [ source 0 target 2]
edge [ source 1 target 3]
edge [ source 3 target 4]
edge [ source 4 target 5]
edge [ source 5 target 6]
edge [ source 5 target 7]

edge [ source 1 target 18]
edge [ source 3 target 8]
edge [ source 4 target 9]
edge [ source 4 target 10]
edge [ source 5 target 11]
edge [ source 6 target 12]
edge [ source 6 target 13]
edge [ source 6 target 14]
edge [ source 7 target 15]
edge [ source 7 target 16]
edge [ source 7 target 17]


]
