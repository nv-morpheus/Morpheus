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

#include "morpheus/io/data_loader.hpp"

#include <mrc/modules/properties/persistent.hpp>
#include <mrc/modules/segment_modules.hpp>
#include <nlohmann/json.hpp>

namespace morpheus {
#pragma GCC visibility push(default)
class DataLoaderModule : public mrc::modules::SegmentModule, public mrc::modules::PersistentModule
{
    using type_t = DataLoaderModule;

  public:
    DataLoaderModule(std::string module_name);
    DataLoaderModule(std::string module_name, nlohmann::json config);

  protected:
    void initialize(mrc::segment::Builder& builder) override;
    std::string module_type_name() const override;

  private:
    DataLoader m_data_loader{};
};
#pragma GCC visibility pop
}  // namespace morpheus