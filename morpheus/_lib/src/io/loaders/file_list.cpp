/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/io/loaders/file_list.hpp"

#include <boost/filesystem.hpp>

#include <memory>

namespace morpheus {
std::shared_ptr<MessageControl> FileListLoader::load(std::shared_ptr<MessageControl> control_message)
{
    VLOG(30) << "Called FileListLoader::load()";

    auto config = control_message->config();
    if (!config.contains("directories"))
    {
        throw std::runtime_error("FileListLoader: No directories specified in config");
    }

    auto files       = nlohmann::json::array();
    auto directories = config["directories"];
    for (auto& directory : directories)
    {
        auto dirpath = boost::filesystem::path(directory);
        if (!boost::filesystem::is_directory(dirpath))
        {
            throw std::runtime_error("FileListLoader: " + directory.get<std::string>() + " is not a directory");
        }

        for (boost::filesystem::directory_iterator itr(dirpath); itr != boost::filesystem::directory_iterator(); ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->path()))
            {
                auto filename = itr->path().filename().string();
                VLOG(30) << "FileListLoader: Found file: " << filename;

                files.push_back(filename);
            }
        }
    }

    // TODO(Devin): Improve robustness
    // For now, a directory listing will just create an updated control message with the new file list
    // Should this be a data frame payload instead?
    config["data"] = {{"loader_id", "file"}, {"properties", {"files", files}}};

    return control_message;
}
}  // namespace morpheus