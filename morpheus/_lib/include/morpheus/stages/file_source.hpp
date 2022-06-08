/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <morpheus/messages/meta.hpp>

#include <pysrf/node.hpp>
#include <srf/segment/builder.hpp>

#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** FileSourceStage*************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class FileSourceStage : public srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    FileSourceStage(std::string filename, int repeat = 1);

  private:
    subscriber_fn_t build();
    /**
     * TODO(Documentation)
     */
    cudf::io::table_with_metadata load_table();

    std::string m_filename;
    int m_repeat{1};
};

/****** FileSourceStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct FileSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a FileSourceStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<FileSourceStage>> init(srf::segment::Builder &parent,
                                                                       const std::string &name,
                                                                       std::string filename,
                                                                       int repeat = 1);
};
#pragma GCC visibility pop
}  // namespace morpheus
