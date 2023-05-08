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

import cudf

import morpheus.parsers.ip as ip


def test_ip_to_int():
    input = cudf.Series(["5.79.97.178", "94.130.74.45"])
    expected = cudf.Series([89088434, 1585596973])
    actual = ip.ip_to_int(input)
    assert actual.equals(expected)


def test_int_to_ip():
    input = cudf.Series([89088434, 1585596973])
    expected = cudf.Series(["5.79.97.178", "94.130.74.45"])
    actual = ip.int_to_ip(input)
    assert actual.equals(expected)


def test_is_ip():
    input = cudf.Series(["5.79.97.178", "1.2.3.4", "5", "5.79", "5.79.97", "5.79.97.178.100"])
    expected = cudf.Series([True, True, False, False, False, False])
    actual = ip.is_ip(input)
    assert actual.equals(expected)


def test_is_reserved():
    input = cudf.Series(["240.0.0.0", "255.255.255.255", "5.79.97.178"])
    expected = cudf.Series([True, True, False])
    actual = ip.is_reserved(input)
    assert actual.equals(expected)


def test_is_loopback():
    input = cudf.Series(["127.0.0.1", "5.79.97.178"])
    expected = cudf.Series([True, False])
    actual = ip.is_loopback(input)
    assert actual.equals(expected)


def test_is_link_local():
    input = cudf.Series(["169.254.0.0", "5.79.97.178"])
    expected = cudf.Series([True, False])
    actual = ip.is_link_local(input)
    assert actual.equals(expected)


def test_is_unspecified():
    input = cudf.Series(["0.0.0.0", "5.79.97.178"])
    expected = cudf.Series([True, False])
    actual = ip.is_unspecified(input)
    assert actual.equals(expected)


def test_is_multicast():
    input = cudf.Series(["224.0.0.0", "239.255.255.255", "5.79.97.178"])
    expected = cudf.Series([True, True, False])
    actual = ip.is_multicast(input)
    assert actual.equals(expected)


def test_is_private():
    input = cudf.Series(["0.0.0.0", "5.79.97.178"])
    expected = cudf.Series([True, False])
    actual = ip.is_private(input)
    assert actual.equals(expected)


def test_is_global():
    input = cudf.Series(["0.0.0.0", "5.79.97.178"])
    expected = cudf.Series([False, True])
    actual = ip.is_global(input)
    assert actual.equals(expected)


def test_netmask():
    input = cudf.Series(["5.79.97.178", "94.130.74.45"])
    expected = cudf.Series(["255.255.128.0", "255.255.128.0"])
    actual = ip.netmask(input, 17)
    assert actual.equals(expected)


def test_hostmask():
    input = cudf.Series(["5.79.97.178", "94.130.74.45"])
    expected = cudf.Series(["0.0.127.255", "0.0.127.255"])
    actual = ip.hostmask(input, 17)
    assert actual.equals(expected)


def test_mask():
    input_ips = cudf.Series(["5.79.97.178", "94.130.74.45"])
    input_masks = cudf.Series(["255.255.128.0", "255.255.128.0"])
    expected = cudf.Series(["5.79.0.0", "94.130.0.0"])
    actual = ip.mask(input_ips, input_masks)
    assert actual.equals(expected)
