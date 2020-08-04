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


import numpy as np
from vermouth.processors.molecule_grouper import constrained_kmeans

def test_constrained_kmeans():
    data = np.random.random((100, 3))
    print(constrained_kmeans(data, 25, init_clusters='random', precision=100))
    assert False


test_constrained_kmeans()