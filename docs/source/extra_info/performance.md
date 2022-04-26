<!--
SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Performance
* Section will discuss everything related to performance from the users perspective
  * This is higher level and more configuration focused than what a Morpheus developer might think when discussing performance
* Should cover topics such as:
  * Python stages and why they are bad
  * The evil GIL and ways to avoid it
    * Discuss things like cython and numba `nogil`
  * Choosing the right batch size
  * Choosing the right edge buffer size
  * Choosing the right number of threads
* Dedicated section to measuring and recording telemetry
  * Cover the monitor stage and its pitfalls
