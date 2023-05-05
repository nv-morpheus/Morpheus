# Copyright (c) 2023, NVIDIA CORPORATION.
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

import multiprocessing as mp
import os
import typing

import fsspec
import pandas as pd

VALID_VALUES = frozenset(["single_thread", "multiprocess", "multiprocessing", "dask", "dask_thread"])


class Downloader:
    """
    Downloads a list of `fsspec.core.OpenFiles` files using one of the following methods:
        single_thread, multiprocess, dask or dask_thread

    The download method can be passed in via the `download_method` parameter or via the `MORPHEUS_FILE_DOWNLOAD_TYPE`
    environment variable. If both are set, the environment variable takes precedence, by default `dask_thread` is used.

    Parameters
    ----------
    download_method : str, optional, default = "dask_thread"
        The download method to use, if the `MORPHEUS_FILE_DOWNLOAD_TYPE` environment variable is set, it takes
        presedence.
    """

    def __init__(self, download_method: str = "dask_thread"):
        self._download_method = os.environ.get("MORPHEUS_FILE_DOWNLOAD_TYPE", download_method)

        if self._download_method not in VALID_VALUES:
            raise ValueError(f"Invalid download method: {self._download_method}. Valid values are: {VALID_VALUES}")

    @property
    def download_method(self) -> str:
        return self._download_method

    def download(self,
                 download_buckets: typing.Iterable[fsspec.core.OpenFiles],
                 download_fn: typing.Callable[[fsspec.core.OpenFiles], pd.DataFrame],
                 get_dask_client_fn: typing.Callable = None) -> typing.List[pd.DataFrame]:
        """
        Download the files in `download_buckets` using the method specified in the constructor.
        If dask or dask_thread is used, the `get_dask_client_fn` function is used to create a dask client, otherwise
        it is not called. If using one of the other methods this can be set to None.

        Parameters
        ----------
        download_buckets : typing.Iterable[fsspec.core.OpenFiles]
            Files to download
        download_fn : typing.Callable[[fsspec.core.OpenFiles], pd.DataFrame]
            Function used to download an individual file and return the contents as a pandas DataFrame
        get_dask_client_fn : typing.Callable[[], dask.distributed.Client])
            Function used to instantiate a dask client. Only used if the download method is dask or dask_thread.

        Returns
        -------
        typing.List[pd.DataFrame]
        """
        dfs = []
        if (self._download_method.startswith("dask")):
            assert get_dask_client_fn is not None, "get_dask_client_fn must be set if using dask or dask_thread"

            # Create the client each time to ensure all connections to the cluster are closed (they can time out)
            with get_dask_client_fn() as client:
                dfs = client.map(download_fn, download_buckets)

                dfs = client.gather(dfs)

        elif (self._download_method in ("multiprocess", "multiprocessing")):
            # Use multiprocessing here since parallel downloads are a pain
            with mp.get_context("spawn").Pool(mp.cpu_count()) as p:
                dfs = p.map(download_fn, download_buckets)
        else:
            # Simply loop
            for open_file in download_buckets:
                dfs.append(download_fn(open_file))

        return dfs
