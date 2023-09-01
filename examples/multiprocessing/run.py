# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import multiprocessing as mp
import threading as mt
from multiprocessing.connection import Connection

import mrc
from mrc.core.subscriber import Observer

import cudf
import time
import os

from morpheus.config import Config
from morpheus.messages.multi_message import MultiMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


class MyMultiprocessingStage(SinglePortStage):

    @property
    def name(self) -> str:
        return "my-multiprocessing"

    def accepted_types(self) -> tuple:
        return (MultiMessage, )

    def supports_cpp_node(self):
        return False

    @staticmethod
    def generate(obs: mrc.Observable, sub: mrc.Subscriber):

        # generate is called on subscribe

        def child_receive(conn: Connection):
            print("===== Started child receive =====", os.getppid(), os.getpid())
            while True:
                while not conn.poll():
                    print("child: waiting...")
                    time.sleep(10)
                print("child: receiving...")
                message: MultiMessage = conn.recv()
                print("received something in child!")
                conn.send(message)

        def parent_receive(conn: Connection):
            print("===== Started parent receive =====", os.getpid())
            while True:
                while not conn.poll():
                    print("parent: waiting...")
                    time.sleep(1)
                message: MultiMessage = conn.recv()
                sub.on_next(message)

        [parent_conn, child_conn] = mp.Pipe()

        my_process = mp.Process(target=child_receive, args=(child_conn,))
        my_thread = mt.Thread(target=parent_receive, args=(parent_conn,))

        def on_next(message: MultiMessage):
            print("obs on next", os.getpid())
            try:
                parent_conn.send(cudf.DataFrame({"a": "b"}))
            except Exception as error:
                print(error) # can't pickle a MultiMessage
                my_process.kill()
                my_thread.kill()

        def on_error(error: BaseException):
            print("obs on error", os.getpid())
            my_process.kill()
            my_thread.kill()
            sub.on_error(error)


        def on_completed():
            print("obs on completed", os.getpid())
            # my_process.kill()
            # my_thread.kill()
            # sub.on_completed()

        # forward the subscribe call

        my_process.start()
        my_thread.start()

        obs.subscribe(Observer.make_observer(on_next, on_error, on_completed))

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair):
        stream = builder.make_node("my-multiprocessing", mrc.core.operators.build(MyMultiprocessingStage.generate))
        builder.make_edge(input_stream[0], stream)
        return stream, input_stream[1]


def run_pipeline():

    config = Config()

    df_input = cudf.DataFrame({"name": "1", "value": 1})

    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [df_input]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(MyMultiprocessingStage(config))
    pipeline.add_stage(SerializeStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    messages = sink.get_messages()

    if len(messages) > 0:
        print(messages[0].copy_dataframe())


if __name__ == "__main__":
    run_pipeline()
