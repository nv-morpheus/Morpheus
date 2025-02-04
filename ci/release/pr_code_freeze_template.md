<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## :snowflake: Code freeze for `branch-${VERSION}` and `v${VERSION}` release

### What does this mean?
Only critical/hotfix level issues should be merged into `branch-${VERSION}` until release (merging of this PR).

All other development PRs should be retargeted towards the next release branch: `branch-${NEXT_VERSION}`.

### What is the purpose of this PR?
- Update documentation
- Allow testing for the new release
- Enable a means to merge `branch-${VERSION}` into `main` for the release
