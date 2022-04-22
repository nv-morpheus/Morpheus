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

#include <morpheus/utilities/matx_util.hpp>

#include <morpheus/objects/dev_mem_info.hpp>
#include <morpheus/utilities/type_util.hpp>

#include <neo/cuda/sync.hpp>
#include <neo/core/tensor.hpp>

#include <matx.h>

namespace morpheus {

    // Component-private classes.
    // ************ MatxUtil__MatxCast**************//
    /**
     * TODO(Documentation)
     */
    struct MatxUtil__MatxCast { // NOLINT
        size_t element_count;
        rmm::cuda_stream_view stream;

        /**
         * TODO(Documentation)
         */
        template<typename InputT,
                typename OutputT,
                std::enable_if_t<!cudf::is_numeric<InputT>() || !cudf::is_numeric<OutputT>()> * = nullptr>
        void operator()(void *input_data, void *output_data) {
            throw std::invalid_argument("Unsupported conversion");
        }

        /**
         * TODO(Documentation)
         */
        template<typename InputT,
                typename OutputT,
                std::enable_if_t<cudf::is_numeric<InputT>() && cudf::is_numeric<OutputT>()> * = nullptr>
        void operator()(void *input_data, void *output_data) {
            matx::tensorShape_t<1> shape({static_cast<matx::index_t>(element_count)});

            matx::tensor_t<InputT, 1> input_tensor(static_cast<InputT *>(input_data), shape);
            matx::tensor_t<OutputT, 1> output_tensor(static_cast<OutputT *>(output_data), shape);

            (output_tensor = input_tensor).run(stream.value());
        }
    };

    // ************ MatxUtil__MatxCreateSegIds**************//
    /**
     * TODO(Documentation)
     */
    struct MatxUtil__MatxCreateSegIds {
        size_t element_count;
        size_t fea_len;
        rmm::cuda_stream_view stream;

        /**
         * TODO(Documentation)
         */
        template<typename OutputT, std::enable_if_t<!std::is_integral_v<OutputT>> * = nullptr>
        void operator()(void *output_data) {
            throw std::invalid_argument("Unsupported conversion");
        }

        /**
         * TODO(Documentation)
         */
        template<typename OutputT, std::enable_if_t<std::is_integral_v<OutputT>> * = nullptr>
        void operator()(void *output_data) {
            matx::tensorShape_t<2> shape({static_cast<matx::index_t>(element_count), 3});

            matx::tensor_t<OutputT, 2> output_tensor(static_cast<OutputT *>(output_data), shape);

            auto col0 = output_tensor.template Slice<1>({0, 0}, {matx::matxEnd, matx::matxDropDim});
            auto col2 = output_tensor.template Slice<1>({0, 2}, {matx::matxEnd, matx::matxDropDim});
            auto range_col =
                    matx::range_x<OutputT>(matx::tensorShape_t<1>({static_cast<matx::index_t>(element_count)}), 0, 1);

            (col0 = range_col).run(stream.value());
            (col2 = fea_len - 1).run(stream.value());
        }
    };  // NOLINT

    // ************ MatxUtil__MatxLogits**************//
    /**
     * TODO(Documentation)
     */
    struct MatxUtil__MatxLogits { // NOLINT
        size_t element_count;
        rmm::cuda_stream_view stream;

        /**
         * TODO(Documentation)
         */
        template<typename InputT, std::enable_if_t<!cudf::is_floating_point<InputT>()> * = nullptr>
        void operator()(void *input_data, void *output_data) {
            throw std::invalid_argument("Unsupported conversion");
        }

        /**
         * TODO(Documentation)
         */
        template<typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()> * = nullptr>
        void operator()(void *input_data, void *output_data) {
            matx::tensorShape_t<1> shape({static_cast<matx::index_t>(element_count)});

            matx::tensor_t<InputT, 1> input_tensor(static_cast<InputT *>(input_data), shape);

            matx::tensor_t<InputT, 1> output_tensor(static_cast<InputT *>(output_data), shape);

            (output_tensor = (InputT) 1 / ((InputT) 1 + matx::exp((InputT) -1 * input_tensor))).run(stream.value());
        }
    }; // NOLINT

    // ************ MatxUtil__MatxTranspose**************//
    /**
     * TODO(Documentation)
     */
    struct MatxUtil__MatxTranspose { // NOLINT
        size_t element_count;
        rmm::cuda_stream_view stream;
        size_t rows;
        size_t cols;

        /**
         * TODO(Documentation)
         */
        template<typename InputT, std::enable_if_t<!cudf::is_numeric<InputT>()> * = nullptr>
        void operator()(void *input_data, void *output_data) {
            throw std::invalid_argument("Unsupported conversion");
        }

        /**
         * TODO(Documentation)
         */
        template<typename InputT, std::enable_if_t<cudf::is_numeric<InputT>()> * = nullptr>
        void operator()(void *input_data, void *output_data) {
            matx::tensorShape_t<2> input_shape({static_cast<matx::index_t>(rows), static_cast<matx::index_t>(cols)});
            matx::tensorShape_t<2> output_shape({static_cast<matx::index_t>(cols), static_cast<matx::index_t>(rows)});

            matx::tensor_t<InputT, 2> input_tensor(static_cast<InputT *>(input_data), input_shape);
            matx::tensor_t<InputT, 2> output_tensor(static_cast<InputT *>(output_data), output_shape);

            (output_tensor = input_tensor.Permute({1, 0})).run(stream.value());
        }
    };

    // ************ MatxUtil__MatxThreshold**************//
    /**
     * TODO(Documentation)
     */
    struct MatxUtil__MatxThreshold { // NOLINT
        size_t rows;
        size_t cols;
        bool by_row;
        rmm::cuda_stream_view stream;

        /**
         * TODO(Documentation)
         */
        template<typename InputT, std::enable_if_t<!cudf::is_floating_point<InputT>()> * = nullptr>
        void
        operator()(void *input_data, void *output_data, double threshold, const std::vector<neo::TensorIndex> &stride) {
            throw std::invalid_argument("Unsupported conversion");
        }

        /**
         * TODO(Documentation)
         */
        template<typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()> * = nullptr>
        void
        operator()(void *input_data, void *output_data, double threshold, const std::vector<neo::TensorIndex> &stride) {
            if (by_row) {
                this->threshold_by_row<InputT>(input_data, output_data, threshold, stride);
            } else {
                this->threshold<InputT>(input_data, output_data, threshold, stride);
            }
        }

    private:
        /**
         * TODO(Documentation)
         */
        template<typename InputT>
        void threshold_by_row(void *input_data, void *output_data, double threshold,
                              const std::vector<neo::TensorIndex> &stride) {
            matx::tensorShape_t<2> input_shape({static_cast<matx::index_t>(rows), static_cast<matx::index_t>(cols)});

            // Output is always 1 column
            matx::tensorShape_t<1> output_shape({static_cast<matx::index_t>(rows)});

            // Specify the stride here since the data comes in column major order.
            matx::tensor_t<InputT, 2> input_tensor(static_cast<InputT *>(input_data),
                                                   input_shape,
                                                   {static_cast<matx::index_t>(stride[0]),
                                                    static_cast<matx::index_t>(stride[1])});

            // Tmp array to hold max value
            matx::tensor_t<InputT, 1> max_tensor(output_shape);

            // row-wise reduction
            matx::rmax(max_tensor, input_tensor, stream.value());

            matx::tensor_t<bool, 1> output_tensor(static_cast<bool *>(output_data), output_shape);

            // Convert max value to bool
            (output_tensor = max_tensor > (InputT) threshold).run(stream.value());
        }

        /**
         * TODO(Documentation)
         */
        template<typename InputT>
        void
        threshold(void *input_data, void *output_data, double threshold, const std::vector<neo::TensorIndex> &stride) {
            matx::tensorShape_t<2> shape({static_cast<matx::index_t>(rows), static_cast<matx::index_t>(cols)});

            matx::index_t matx_stride[2] = {static_cast<matx::index_t>(stride[0]),
                                            static_cast<matx::index_t>(stride[1])};

            matx::tensor_t<InputT, 2> input_tensor(static_cast<InputT *>(input_data), shape, matx_stride);
            matx::tensor_t<bool, 2> output_tensor(static_cast<bool *>(output_data), shape, matx_stride);

            // Convert max value to bool
            (output_tensor = input_tensor > (InputT) threshold).run(stream.value());
        }
    };

    // Component public implementations
    // ************ MatxUtil************************* //
    std::shared_ptr<rmm::device_buffer> MatxUtil::cast(const DevMemInfo &input, neo::TypeId output_type) {
        auto input_dtype = DType(input.type_id);
        auto output_dtype = DType(output_type);

        // Create the output
        auto output = std::make_shared<rmm::device_buffer>(
                output_dtype.item_size() * input.element_count, input.buffer->stream(),
                input.buffer->memory_resource());

        cudf::double_type_dispatcher(cudf::data_type{input_dtype.cudf_type_id()},
                                     cudf::data_type{output_dtype.cudf_type_id()},
                                     MatxUtil__MatxCast{input.element_count, output->stream()},
                                     input.data(),
                                     output->data());

        neo::enqueue_stream_sync_event(output->stream()).get();

        return output;
    }

    std::shared_ptr<rmm::device_buffer>
    MatxUtil::create_seg_ids(size_t row_count, size_t fea_len, neo::TypeId output_type) {
        auto output_dtype = DType(output_type);

        // Now create the output
        auto output =
                std::make_shared<rmm::device_buffer>(output_dtype.item_size() * row_count * 3,
                                                     rmm::cuda_stream_per_thread);

        cudf::type_dispatcher(cudf::data_type{output_dtype.cudf_type_id()},
                              MatxUtil__MatxCreateSegIds{row_count, fea_len, output->stream()},
                              output->data());

        return output;
    }

    std::shared_ptr<rmm::device_buffer> MatxUtil::logits(const DevMemInfo &input) {
        auto input_dtype = DType(input.type_id);

        // Now create the output
        auto output = std::make_shared<rmm::device_buffer>(
                input_dtype.item_size() * input.element_count, input.buffer->stream(), input.buffer->memory_resource());

        cudf::type_dispatcher(cudf::data_type{input_dtype.cudf_type_id()},
                              MatxUtil__MatxLogits{input.element_count, output->stream()},
                              input.data(),
                              output->data());

        return output;
    }

    std::shared_ptr<rmm::device_buffer> MatxUtil::transpose(const DevMemInfo &input, size_t rows, size_t cols) {
        auto input_dtype = DType(input.type_id);

        // Now create the output
        auto output = std::make_shared<rmm::device_buffer>(
                input_dtype.item_size() * input.element_count, input.buffer->stream(), input.buffer->memory_resource());

        cudf::type_dispatcher(cudf::data_type{input_dtype.cudf_type_id()},
                              MatxUtil__MatxTranspose{input.element_count, output->stream(), rows, cols},
                              input.data(),
                              output->data());

        return output;
    }

    std::shared_ptr<rmm::device_buffer>
    MatxUtil::threshold(const DevMemInfo &input, size_t rows, size_t cols,
                        const std::vector<neo::TensorIndex> &stride,
                        double thresh_val, bool by_row) {
        auto input_dtype = DType(input.type_id);

        std::size_t output_size = sizeof(bool) * rows;
        if (!by_row) {
            output_size *= cols;
        }

        // Now create the output array of bools
        auto output = std::make_shared<rmm::device_buffer>(output_size, input.buffer->stream(),
                                                           input.buffer->memory_resource());

        cudf::type_dispatcher(cudf::data_type{input_dtype.cudf_type_id()},
                              MatxUtil__MatxThreshold{rows, cols, by_row, output->stream()},
                              input.data(),
                              output->data(),
                              thresh_val,
                              stride);

        neo::enqueue_stream_sync_event(output->stream()).get();

        return output;
    }
}