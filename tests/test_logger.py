# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import multiprocessing
import os
from unittest.mock import patch

import pytest

from _utils import TEST_DIRS
from morpheus.utils.logger import TqdmLoggingHandler
from morpheus.utils.logger import configure_logging
from morpheus.utils.logger import deprecated_message_warning
from morpheus.utils.logger import deprecated_stage_warning
from morpheus.utils.logger import set_log_level


@patch('logging.handlers.QueueListener')
@patch('logging.handlers.QueueHandler.emit')
def test_configure_logging_from_level_default_handlers(queue_handler, queue_listener):
    configure_logging(log_level=logging.DEBUG)
    morpheus_logger = logging.getLogger("morpheus")
    assert morpheus_logger.level == logging.DEBUG
    assert morpheus_logger.propagate is False
    assert isinstance(morpheus_logger.handlers[1].queue, multiprocessing.queues.Queue)
    pos_args = queue_listener.call_args[0]
    assert len(pos_args) == 3
    assert isinstance(pos_args[0], multiprocessing.queues.Queue)
    assert isinstance(pos_args[1], TqdmLoggingHandler)
    assert isinstance(pos_args[2], logging.handlers.RotatingFileHandler)
    assert pos_args[2].baseFilename.endswith("morpheus.log")
    morpheus_logger.debug("test")
    queue_handler.assert_called()


def test_configure_logging__no_args():
    with pytest.raises(Exception) as excinfo:
        configure_logging()
    assert "log_level must be specified" in str(excinfo.value)


@patch('logging.handlers.RotatingFileHandler.emit')
@patch('morpheus.utils.logger.TqdmLoggingHandler.emit')
def test_configure_logging_from_file(console_handler, file_handler):
    log_config_file = os.path.join(TEST_DIRS.tests_data_dir, "logging.json")
    configure_logging(log_config_file=log_config_file)
    morpheus_logger = logging.getLogger("morpheus")
    assert morpheus_logger.level == logging.DEBUG
    assert morpheus_logger.propagate is False
    morpheus_logger.debug("test")
    console_handler.assert_called_once()
    file_handler.assert_called_once()


def test_configure_logging_from_file_filenotfound():
    with pytest.raises(FileNotFoundError):
        configure_logging(log_config_file="does_not_exist.json")


@patch('logging.handlers.QueueListener')
@patch('logging.handlers.QueueHandler.emit')
def test_configure_logging_add_one_handler(queue_handler, queue_listener):
    new_handler = logging.StreamHandler()
    configure_logging(new_handler, log_level=logging.DEBUG)
    morpheus_logger = logging.getLogger("morpheus")
    assert morpheus_logger.level == logging.DEBUG
    assert morpheus_logger.propagate is False
    pos_args = queue_listener.call_args[0]
    assert len(pos_args) == 4
    assert isinstance(pos_args[0], multiprocessing.queues.Queue)
    assert isinstance(pos_args[1], TqdmLoggingHandler)
    assert isinstance(pos_args[2], logging.handlers.RotatingFileHandler)
    assert isinstance(pos_args[3], logging.StreamHandler)
    morpheus_logger.debug("test")
    queue_handler.assert_called()


@patch('logging.handlers.QueueListener')
@patch('logging.handlers.QueueHandler.emit')
def test_configure_logging_add_two_handlers(queue_handler, queue_listener):
    new_handler_1 = logging.StreamHandler()
    new_handler_2 = logging.StreamHandler()
    configure_logging(new_handler_1, new_handler_2, log_level=logging.DEBUG)
    morpheus_logger = logging.getLogger("morpheus")
    assert morpheus_logger.level == logging.DEBUG
    assert morpheus_logger.propagate is False
    pos_args = queue_listener.call_args[0]
    assert len(pos_args) == 5
    assert isinstance(pos_args[0], multiprocessing.queues.Queue)
    assert isinstance(pos_args[1], TqdmLoggingHandler)
    assert isinstance(pos_args[2], logging.handlers.RotatingFileHandler)
    assert isinstance(pos_args[3], logging.StreamHandler)
    assert isinstance(pos_args[4], logging.StreamHandler)
    morpheus_logger.debug("test")
    queue_handler.assert_called()


def test_set_log_level():
    configure_logging(log_level=logging.INFO)
    morpheus_logger = logging.getLogger("morpheus")
    assert morpheus_logger.level == logging.INFO
    set_log_level(logging.DEBUG)
    assert morpheus_logger.level == logging.DEBUG


def test_deprecated_stage_warning(caplog):

    class DummyStage():
        pass

    logger = logging.getLogger()
    caplog.set_level(logging.WARNING)
    deprecated_stage_warning(logger, DummyStage, "dummy_stage")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "The 'DummyStage' stage ('dummy_stage') has been deprecated" in caplog.text


def test_deprecated_stage_warning_with_reason(caplog):

    class DummyStage():
        pass

    logger = logging.getLogger()
    caplog.set_level(logging.WARNING)
    deprecated_stage_warning(logger, DummyStage, "dummy_stage", reason="This is the reason.")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "The 'DummyStage' stage ('dummy_stage') has been deprecated and will be removed in a future version. " \
           "This is the reason." in caplog.text


def test_deprecated_message_warning(caplog):

    class OldMessage():
        pass

    class NewMessage():
        pass

    logger = logging.getLogger()
    caplog.set_level(logging.WARNING)
    deprecated_message_warning(logger, OldMessage, NewMessage)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "The 'OldMessage' message has been deprecated and will be removed in a future version. " \
           "Please use 'NewMessage' instead." in caplog.text
