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

#include <morpheus/utilities/matx_util.hpp>
#include <morpheus/utilities/type_util.hpp>

#include <neo/core/tensor.hpp>
#include <pyneo/node.hpp>

#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace morpheus {
    /****** Component public implementations *******************/
    /****** RMMTensor****************************************/
    /**
     * TODO(Documentation)
     */
    class RMMTensor : public neo::ITensor {
    public:
        RMMTensor(std::shared_ptr<rmm::device_buffer> device_buffer,
                  size_t offset,
                  DType dtype,
                  std::vector<neo::TensorIndex> shape,
                  std::vector<neo::TensorIndex> stride = {});

        ~RMMTensor() = default;

        /**
         * TODO(Documentation)
         */
        bool is_compact() const final;

        /**
         * TODO(Documentation)
         */
        neo::DataType dtype() const override;

        /**
         * TODO(Documentation)
         */
        neo::RankType rank() const final;

        /**
         * TODO(Documentation)
         */
        std::shared_ptr<neo::ITensor> deep_copy() const override;

        /**
         * TODO(Documentation)
         */
        std::shared_ptr<neo::ITensor> reshape(const std::vector<neo::TensorIndex> &dims) const override;

        /**
         * TODO(Documentation)
         */
        std::shared_ptr<neo::ITensor> slice(const std::vector<neo::TensorIndex> &min_dims,
                                            const std::vector<neo::TensorIndex> &max_dims) const override;

        /**
         * TODO(Documentation)
         */
        std::shared_ptr<neo::MemoryDescriptor> get_memory() const override;

        /**
         * TODO(Documentation)
         */
        std::size_t bytes() const final;

        /**
         * TODO(Documentation)
         */
        std::size_t count() const final;

        /**
         * TODO(Documentation)
         */
        std::size_t shape(std::size_t idx) const final;

        /**
         * TODO(Documentation)
         */
        std::size_t stride(std::size_t idx) const final;

        /**
         * TODO(Documentation)
         */
        void *data() const override;

        /**
         * TODO(Documentation)
         */
        void get_shape(std::vector<neo::TensorIndex> &s) const;

        /**
         * TODO(Documentation)
         */
        void get_stride(std::vector<neo::TensorIndex> &s) const;

        // Tensor reshape(std::vector<neo::TensorIndex> shape)
        // {
        //     CHECK(is_compact());
        //     return Tensor(descriptor_shared(), dtype_size(), shape);
        // }

        /**
         * TODO(Documentation)
         */
        std::shared_ptr<ITensor> as_type(neo::DataType dtype) const override;

    protected:
    private:
        /**
         * TODO(Documentation)
         */
        size_t offset_bytes() const;

        // Memory info
        std::shared_ptr<rmm::device_buffer> m_md;
        size_t m_offset;

        // // Type info
        // std::string m_typestr;
        // std::size_t m_dtype_size;
        DType m_dtype;

        // Shape info
        std::vector<neo::TensorIndex> m_shape;
        std::vector<neo::TensorIndex> m_stride;
    };
}