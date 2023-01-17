/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/meta.hpp"

#include <cudf/io/types.hpp>               // for table_with_metadata
#include <mrc/channel/status.hpp>          // for Status
#include <mrc/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>  // for Object
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity

#include <memory>
#include <string>
#include <vector>  // for vector

namespace morpheus {
/****** Component public implementations *******************/
/****** FileSourceStage*************************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief Load messages from a file. Source stage is used to load messages from a file and
 * dumping the contents into the pipeline immediately. Useful for testing performance and accuracy of a pipeline.
 */
class FileSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    /**
     * @brief Construct a new File Source Stage object
     *
     * @param filename : Name of the file from which the messages will be read
     * @param repeat : Repeats the input dataset multiple times. Useful to extend small datasets for debugging
     */
    FileSourceStage(std::string filename, int repeat = 1);

  private:
    subscriber_fn_t build();

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
     * @brief Create and initialize a FileSourceStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param filename : Name of the file from which the messages will be read.
     * @param repeat : Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
     * @return std::shared_ptr<mrc::segment::Object<FileSourceStage>>
     */
    static std::shared_ptr<mrc::segment::Object<FileSourceStage>> init(mrc::segment::Builder& builder,
                                                                       const std::string& name,
                                                                       std::string filename,
                                                                       int repeat = 1);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
