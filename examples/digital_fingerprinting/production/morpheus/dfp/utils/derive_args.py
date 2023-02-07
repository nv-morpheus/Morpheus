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

import logging
import os
import pickle
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mlflow
import pandas as pd

from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import load_labels_file
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.utils.logger import configure_logging

logger = logging.getLogger(__name__)


class DeriveArgs:

    def __init__(self,
                 skip_user: str,
                 only_user: str,
                 start_time: str,
                 duration: str,
                 log_level: str,
                 cache_dir: str,
                 source: str,
                 tracking_uri: str,
                 train_users: str = None):

        self._skip_users = list(skip_user)
        self._only_users = list(only_user)
        self._start_time = start_time
        self._duration = duration
        self._log_level = log_level
        self._train_users = train_users
        self._cache_dir = cache_dir
        self._include_generic = None
        self._include_individual = None
        self._initialized = False
        self._tracking_uri = tracking_uri
        self._model_name_formatter = "DFP-%s-{user_id}" % (source)
        self._experiment_name_formatter = "dfp/%s/training/{reg_model_name}" % (source)
        self._is_training = (train_users is not None and train_users != "none")

    def verify_init(func):

        def wrapper(self, *args, **kwargs):
            if not self._initialized:
                raise Exception('Instance not initialized')
            return func(self, *args, **kwargs)

        return wrapper

    def _configure_logging(self):

        configure_logging(log_level=self._log_level)
        logging.getLogger("mlflow").setLevel(self._log_level)

        if (len(self._only_users) > 0 and len(self._only_users) > 0):
            logging.error("Option --skip_user and --only_user are mutually exclusive. Exiting")

        logger.info("Running training pipeline with the following options: ")
        logger.info("Train generic_user: %s", self._include_generic)
        logger.info("Skipping users: %s", self._skip_users)
        logger.info("Start Time: %s", self._start_time)
        logger.info("Duration: %s", self._duration)
        logger.info("Cache Dir: %s", self._cache_dir)

    @property
    @verify_init
    def start_time(self):
        return self._start_time

    @property
    @verify_init
    def end_time(self):
        return self._end_time

    @property
    @verify_init
    def include_generic(self):
        return self._include_generic

    @property
    @verify_init
    def include_individual(self):
        return self._include_individual

    @property
    @verify_init
    def duration(self):
        return self._duration

    @property
    def skip_users(self):
        return self._skip_users

    @property
    def only_users(self):
        return self._only_users

    @property
    def cache_dir(self):
        return self._cache_dir

    @property
    def model_name_formatter(self):
        return self._model_name_formatter

    @property
    def is_training(self):
        return self._is_training

    @property
    def experiment_name_formatter(self):
        return self._experiment_name_formatter

    def _set_include_generic(self):
        self._include_generic = self._train_users == "all" or self._train_users == "generic"

    def _set_include_individual(self):
        self._include_individual = self._train_users != "generic"

    def _update_start_stop_time(self):
        self._duration = timedelta(seconds=pd.Timedelta(self._duration).total_seconds())
        if self._start_time is None:
            self._end_time = datetime.now(tz=timezone.utc)
            self._start_time = self._end_time - self._duration
        else:
            if self._start_time.tzinfo is None:
                self._start_time = self._start_time.replace(tzinfo=timezone.utc)

            self._end_time = self._start_time + self._duration

    def _set_mlflow_tracking_uri(self):
        if self._tracking_uri is None:
            raise ValueError("tracking uri should not be None type.")
        # Initialize ML Flow
        mlflow.set_tracking_uri(self._tracking_uri)
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())

    def init(self):
        self._update_start_stop_time()
        self._set_include_generic()
        self._set_include_individual()
        self._configure_logging()
        self._set_mlflow_tracking_uri()
        self._initialized = True


def get_ae_config(labels_file: str,
                  userid_column_name: str,
                  timestamp_column_name: str,
                  use_cpp: bool = False,
                  num_threads: int = os.cpu_count()):

    config = Config()

    CppConfig.set_should_use_cpp(use_cpp)

    config.num_threads = num_threads

    config.ae = ConfigAutoEncoder()

    config.ae.feature_columns = load_labels_file(get_package_relative_file(labels_file))
    config.ae.userid_column_name = userid_column_name
    config.ae.timestamp_column_name = timestamp_column_name

    return config


def pyobj2str(pyobj, encoding):
    str_val = str(pickle.dumps(pyobj), encoding=encoding)
    return str_val
