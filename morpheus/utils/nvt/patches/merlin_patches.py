# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import merlin.dtypes.aliases as mn
import numpy as np


def patch_numpy_dtype_registry():
    """
    Patches the Merlin dtypes registry to support conversion from Merlin 'struct' dtypes to the equivalent numpy object.
    This is necessary to support pandas conversion of input dataframes containing 'struct' dtypes within an NVT
    operator.

    Until this is fixed upstream, with the mappings added to `merlin/dtypes/mappings/numpy.py`, this patch should be
    used. The function is idempotent, and should be called before any NVT operators are used.
    :return:
    """
    from merlin.dtypes import _dtype_registry

    numpy_dtypes = _dtype_registry.mappings["numpy"].from_merlin_
    if (mn.struct not in numpy_dtypes.keys()):
        numpy_dtypes[mn.struct] = [np.dtype("O"), object]
