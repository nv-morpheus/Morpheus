#!/usr/bin/python
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

import glob
import os

from scapy.all import IP  # pylint: disable=no-name-in-module
from scapy.all import UDP  # pylint: disable=no-name-in-module
from scapy.all import RandShort
from scapy.all import Raw
from scapy.all import send


def main():
    os.chdir("dataset")
    for file in glob.glob("*.txt"):
        with open(file, 'r', encoding='utf-8') as fp:
            while True:
                content = fp.read(1024)
                if not content:
                    break
                pkt = IP(src="192.168.2.28", dst="192.168.2.27") / UDP(sport=RandShort(),
                                                                       dport=5001) / Raw(load=content.encode('utf-8'))
                print(pkt)
                send(pkt, iface="enp202s0f0np0")


if __name__ == "__main__":
    main()
