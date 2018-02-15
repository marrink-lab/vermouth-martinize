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
        atomname "HA1"
    ]
    node [
        id 5
        atomname "HA2"
    ]
    node [
        id 6
        atomname "HN"
    ]
edge [ source 0 target 3]
edge [ source 0 target 2]
edge [ source 1 target 3]
edge [ source 3 target 4]
edge [ source 3 target 5]
edge [ source 1 target 6]

]
