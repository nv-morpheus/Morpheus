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

import typing

import pandas as pd
from common.data_models import FeatureConfig
from common.feature_constants import FeatureConstants as fc


class FeatureExtractor():
    """
    This is a helper class to extract required features for ransomware detection pipeline.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self._config = config
        self._features = None

    def _filter_by_pid_process(self, plugin_dict: typing.Dict[str, pd.DataFrame],
                               pid_process: str) -> typing.Dict[str, pd.DataFrame]:
        """
        This function filter plugins data by pid_process.
        """

        filtered_plugin_dict = {}

        for plugin_name in plugin_dict.keys():
            plugin_df = plugin_dict[plugin_name]
            plugin_df = plugin_df[plugin_df.PID_Process == pid_process]
            filtered_plugin_dict[plugin_name] = plugin_df

        return filtered_plugin_dict

    def _count_double_extension(self, file_paths: typing.List[str]):
        """
        This function counts the amount of double extensions to a common2 type files and
        return the largest double extension.
        """

        count = 0

        for file_path in file_paths:

            file_split_dot = file_path.split('.')

            split_dot = file_split_dot[:-1]

            if len(split_dot) > 1:

                for word_dot in split_dot:

                    if word_dot in self._config.file_extns:
                        count += 1

                        break

        self._features['HANDLES_DOUBLE_EXTENSION'] = count

    def _count_files_different_extension(self, file_paths: typing.List[str]):
        count = 0
        for i in range(len(file_paths)):
            if file_paths[i] != None:
                try:
                    file_current = file_paths[i].lower()
                except:
                    continue
                for j in range(len(file_paths)):
                    if j == i:
                        continue
                    if file_paths[j] != None:
                        try:
                            index_file_start = file_paths[j].lower().find(file_current)
                        except:
                            continue
                        index_file_end = index_file_start + len(file_current) - 1
                        if index_file_start > -1:
                            if len(file_paths[j]) > index_file_end + 1 and file_paths[j][
                                index_file_end + 1] == '.':
                                count += 1
                        split_dot = file_current.lower().split('.')
                        if len(split_dot) > 1:
                            if split_dot[-1] in self._config.file_extns:
                                split_slash = file_current.lower().split('\\')
                                if len(split_slash) > 1:
                                    index_file_start = file_paths[j].lower().find(split_slash[-1])
                                    if index_file_start > 0 and file_paths[j][index_file_start - 1] == '.':
                                        count += 1
        self._features['HANDLES_DIFFERENT_EXTENSION'] = count

    def _extract_threadlist(self, x: pd.DataFrame):
        """
        # Count amount of unique states and wait reasons and thread with state and waitreason:
        # '2'-'Running'
        # '9'-'WrPageIn'
        # '13'-'WrUserRequest'
        # '31'-'WrDispatchInt'
        """

        for state in fc.STATE_LIST:
            state_df = x[x.State == state]
            self._features['THREAD_STATE_' + str(state)] = len(state_df)

        for wait_reason in fc.WAIT_REASON_LIST:
            wait_reason_df = x[x.WaitReason == wait_reason]
            self._features['THREAD_WINDOWS_WAIT_REASON_' + str(wait_reason)] = len(wait_reason_df)

    def _extract_vadinfo(self, x: pd.DataFrame):
        """
        This function extracts vadinfo features about commit charged, vad/vads and
        private memory and memory protection type.
        """

        # CommitCharge - is the total amount of virtual memory of all processes that
        # must be backed by either physical memory or the page file
        # vad - virtual address descriptor
        # vads - virtual address descriptor short
        # private memory - this field refers to committed regions that cannot be shared with other processes.

        vad_size = len(x)
        if vad_size>0:
            private_memory_1_df = x[x.PrivateMemory == 1]
            private_memory_0_df = x[x.PrivateMemory == 0]

            # Count vad, vads and private memory amount
            self._features['VMA_AMOUNT'] = vad_size
            self._features['VMA_WINDOWS_PRIVATE_MEMORY_1'] = len(private_memory_1_df)/vad_size
            self._features['VMA_WINDOWS_PRIVATE_MEMORY_0'] = len(private_memory_0_df)/vad_size

            for protection in fc.PROTECTIONS.keys():
                protection_df = x[x.Protection == protection]
                self._features['VMA_WINDOWS_PROTECTION_' + fc.PROTECTIONS[protection]] = len(protection_df)/vad_size
        else:
            self._features['VMA_AMOUNT'] = 0
            self._features['VMA_WINDOWS_PRIVATE_MEMORY_1'] = 0
            self._features['VMA_WINDOWS_PRIVATE_MEMORY_0'] = 0

            for protection in fc.PROTECTIONS.keys():
                self._features['VMA_WINDOWS_PROTECTION_' + fc.PROTECTIONS[protection]] = 0

        if self._features['VMA_WINDOWS_PROTECTION_PAGE_READWRITE_RATIO'] > 0 and self._features['VMA_WINDOWS_PROTECTION_PAGE_READONLY_RATIO'] > 0:
            self._features['VMA_WINDOWS_PROTECTION_PAGE_READWRITE_READONLY_DEVIDE'] = self._features['VMA_WINDOWS_PROTECTION_PAGE_READWRITE_RATIO']/self._features['VMA_WINDOWS_PROTECTION_PAGE_READONLY_RATIO']
        elif self._features['VMA_WINDOWS_PROTECTION_PAGE_READWRITE_RATIO'] > 0:
            self._features['VMA_WINDOWS_PROTECTION_PAGE_READWRITE_READONLY_DEVIDE'] = 1
        else:
            self._features['VMA_WINDOWS_PROTECTION_PAGE_READWRITE_READONLY_DEVIDE'] = 0

        self._features['VMA_WINDOWS_COMMIT_CHARGE_FULL'] = 0
        if fc.FULL_MEMORY_ADDRESS in pd.unique(x.CommitCharge):
            self._features['VMA_WINDOWS_COMMIT_CHARGE_FULL'] = 1

        cc_wo_full = x[x.CommitCharge != fc.FULL_MEMORY_ADDRESS].CommitCharge
        self._features['VMA_WINDOWS_COMMIT_CHARGE_MAX'] = cc_wo_full.max()
        self._features['VMA_WINDOWS_COMMIT_CHARGE_MEAN'] = cc_wo_full.mean()
        self._features['VMA_WINDOWS_COMMIT_CHARGE_STD'] = cc_wo_full.std(ddof=0)

    def _extract_handles(self, x: pd.DataFrame):
        """
        This function extracts features related to handles such as amount and ratio of each handle type.
        """

        # Count number of handles
        self._features['HANDLES_AMOUNT'] = len(x)

        # Amount of files path in handles files
        file_paths = x.Name.str.lower()#x[x.Type == 'File'].Name.str.lower()

        file_paths = file_paths[(~file_paths.isna()) & (file_paths != '')]

        # Count handles files with double extensions
        self._count_double_extension(file_paths=list(file_paths))

    def _extract_ldrmodules(self, x: pd.DataFrame):
        """
        This function extracts size of the ldrmodules process and it's path.
        """
        self._features['LIBS_AMOUNT'] = len(x)
        self._features['LIB_WINDOWS_SIZE_OFIMAGE'] = 0
        self._features['ldrmodules_df_path'] = ""
        if not x.empty:
            process = x.Process.iloc[0].lower()
            x = x[x.Name.str.contains(process)]
            if not x.empty:
                self._features['LIB_WINDOWS_SIZE_OFIMAGE'] = x.Size.iloc[0]
                self._features['ldrmodules_df_path'] = x.Path.iloc[0]

    def extract_features(self, x: pd.DataFrame, feas_all_zeros: typing.Dict[str, int]) -> pd.DataFrame:
        """
        This function extracts all different ransomware features.

        Parameters
        ----------
        x : `pandas.DataFrame`
            Dataframe with appshield snapshot data.
        feas_all_zeros : typing.Dict[str, int]
            Features with default value (0)
        Returns
        -------
        pandas.DataFrame
            Ransomware features dataframe.
        """

        features_per_pid_process = []

        # Get unique PID_Process for a given snapshot
        pid_processes = list(x["PID_Process"].unique())

        # Filter snapshot data by plugin
        plugin_dict = {plugin: x[x.plugin == plugin] for plugin in self._config.interested_plugins}

        # Filter plugin per pid_process and create features
        for pid_process in pid_processes:

            # Setting default value '0' to all features.
            self._features = dict.fromkeys(feas_all_zeros.keys(), 0)
            fltr_plugin_dict = self._filter_by_pid_process(plugin_dict, pid_process)

            try:
                ldrmodules_df = fltr_plugin_dict['ldrmodules']
                threadlist_df = fltr_plugin_dict['threadlist']
                vadinfo_df = fltr_plugin_dict['vadinfo']
                handles_df = fltr_plugin_dict['handles']

            except KeyError as e:
                raise KeyError('Missing required plugins: %s' % (e))

            # Handles plugin features displays the open handles in a process, use the handles command.
            # This applies to files, registry keys, mutexes, named pipes, events, window stations, desktops, threads,
            # and all other types of securable executive objects.
            self._extract_handles(handles_df)
            # Threadlist plugin features displays the threads that are used by a process.
            self._extract_threadlist(threadlist_df)
            # VadInfo plugin features displays extended information about a process's VAD nodes.
            # In particular, it shows:
            # - The address of the MMVAD structure in kernel memory
            # - The starting and ending virtual addresses in process memory that the MMVAD structure pertains to
            # - The VAD Tag
            # - The VAD flags, control flags, etc
            # - The name of the memory mapped file (if one exists)
            # - The memory protection constant (permissions)
            self._extract_vadinfo(vadinfo_df)
            # LdrModules plugin features displays a process's loaded DLLs. LdrModules detects a dll-hiding or injection
            # kind of activities in a process memory.
            self._extract_ldrmodules(ldrmodules_df)
            # Add process pid
            self._features['pid_process'] = pid_process
            # Add process name
            self._features['process_name'] = ''
            if len(handles_df)>0:
                self._features['process_name'] = handles_df['Process'].iloc[0]

            # Add pid_process features to a list
            features_per_pid_process.append(self._features)

        # Convert list of pid_process features to a dataframe
        features_df = pd.DataFrame.from_dict(features_per_pid_process)
        # Snapshot id is used to determine which snapshot the pid_process belongs to
        features_df['snapshot_id'] = x.snapshot_id.iloc[0]
        # Add timestamp. Here we consider only ldrmodules timestamp for all the entries.
        features_df['timestamp'] = x.timestamp.iloc[0]
        return features_df

    @staticmethod
    def combine_features(x: typing.List[pd.DataFrame]) -> pd.DataFrame:
        """
        This function combines features of multiple snapshots to a single dataframe

        Parameters
        ----------
        x : `typing.List[pd.DataFrame]`
            Features of multiple snapshots.

        Returns
        -------
        pandas.DataFrame
            Ransomware features dataframe.
        """
        return pd.concat(x)
