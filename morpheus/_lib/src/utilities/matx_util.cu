/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/utilities/matx_util.hpp"

#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <srf/cuda/sync.hpp>

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
        operator()(void *input_data, void *output_data, double threshold, const std::vector<TensorIndex> &stride) {
            throw std::invalid_argument("Unsupported conversion");
        }

        /**
         * TODO(Documentation)
         */
        template<typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()> * = nullptr>
        void
        operator()(void *input_data, void *output_data, double threshold, const std::vector<TensorIndex> &stride) {
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
                              const std::vector<TensorIndex> &stride) {
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
        threshold(void *input_data, void *output_data, double threshold, const std::vector<TensorIndex> &stride) {
            matx::tensorShape_t<2> shape({static_cast<matx::index_t>(rows), static_cast<matx::index_t>(cols)});

            matx::index_t matx_stride[2] = {static_cast<matx::index_t>(stride[0]),
                                            static_cast<matx::index_t>(stride[1])};

            matx::tensor_t<InputT, 2> input_tensor(static_cast<InputT *>(input_data), shape, matx_stride);
            matx::tensor_t<bool, 2> output_tensor(static_cast<bool *>(output_data), shape, matx_stride);

            // Convert max value to bool
            (output_tensor = input_tensor > (InputT) threshold).run(stream.value());
        }
    };

    struct MatxUtil__MatxReduceMax {
        matx::index_t num_input_rows;
        matx::index_t num_cols;
        std::vector<matx::index_t> input_stride;
        matx::index_t num_output_rows;
        void *input_data;
        void *output_data;
        rmm::cuda_stream_view stream;

        template<typename InputT, std::enable_if_t<!cudf::is_floating_point<InputT>()> * = nullptr>
        void operator()(std::size_t start, std::size_t stop, int32_t output_idx) {
            throw std::invalid_argument("Unsupported conversion");
        }

        template<typename InputT, std::enable_if_t<cudf::is_floating_point<InputT>()> * = nullptr>
        void operator()(std::size_t start, std::size_t stop, int32_t output_idx) {
            auto input_count = stop - start;
            matx::tensorShape_t<2> input_shape({static_cast<matx::index_t>(input_count), num_cols});
            matx::tensorShape_t<1> output_shape({num_cols});

            matx::index_t output_stride[2] = {input_stride[0], input_stride[1]};
            if (output_stride[0] == 1)
            {
                output_stride[1] = num_output_rows;
            }

            auto input_ptr = static_cast<InputT *>(input_data) + (start * input_stride[0]);
            auto output_ptr = static_cast<InputT *>(output_data) + (output_idx *  output_stride[0]);

            matx::tensor_t<InputT, 2> input_tensor(input_ptr, input_shape, {input_stride[0], input_stride[1]});
            matx::tensor_t<InputT, 1> output_tensor(output_ptr, output_shape, {output_stride[1]});

            // We need to transpose the input such that rmax will reduce the rows
            // Matx performs reductions over the innermost dimensions.
            // see https://nvidia.github.io/MatX/api/reduce.html
            matx::rmax(output_tensor, input_tensor.Permute({1, 0}), stream.value());
        }
    };

    // Component public implementations
    // ************ MatxUtil************************* //
    std::shared_ptr<rmm::device_buffer> MatxUtil::cast(const DevMemInfo &input, TypeId output_type) {
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

        srf::enqueue_stream_sync_event(output->stream()).get();

        return output;
    }

    std::shared_ptr<rmm::device_buffer>
    MatxUtil::create_seg_ids(size_t row_count, size_t fea_len, TypeId output_type) {
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
                        const std::vector<TensorIndex> &stride,
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

        srf::enqueue_stream_sync_event(output->stream()).get();

        return output;
    }

    std::shared_ptr<rmm::device_buffer>
    MatxUtil::reduce_max(const DevMemInfo &input,
                         const std::vector<int32_t> &seq_ids,
                         size_t seq_id_offset,
                         const std::vector<int64_t> &input_shape,
                         const std::vector<int64_t> &input_stride,
                         const std::vector<int64_t> &output_shape)
    {
        auto dtype = DType(input.type_id);
        auto elem_size = dtype.item_size();
        auto cudf_type = cudf::data_type{dtype.cudf_type_id()};
        auto num_input_rows = input_shape[0];
        auto num_input_cols = input_shape[1];

        std::vector<matx::index_t>matx_stride{input_stride[0], input_stride[1]};
        std::size_t output_element_count = output_shape[0] * output_shape[1];
        std::size_t output_buff_size = elem_size * output_element_count;

        DCHECK(output_element_count <= input.element_count) << "Output buffer size should be less than or equal to the input";
        DCHECK(num_input_cols == output_shape[1]) << "Number of input and output columns must match";

        auto output = std::make_shared<rmm::device_buffer>(output_buff_size,
                                                           input.buffer->stream(),
                                                           input.buffer->memory_resource());

        MatxUtil__MatxReduceMax matx_reduce_max{num_input_rows, num_input_cols, matx_stride, output_shape[0], input.data(), output->data(), output->stream()};

        std::size_t start = 0;
        auto output_offset = seq_ids[seq_id_offset];
        for (std::size_t i=0; i < num_input_rows; ++i)
        {
            auto idx = seq_ids[i+seq_id_offset];
            if (idx != seq_ids[start+seq_id_offset])
            {
                cudf::type_dispatcher(cudf_type,
                                      matx_reduce_max,
                                      start,
                                      i,
                                      seq_ids[start+seq_id_offset]-output_offset);
                start = i;
            }
        }

        cudf::type_dispatcher(cudf_type,
                              matx_reduce_max,
                              start,
                              num_input_rows,
                              seq_ids[start+seq_id_offset]-output_offset);

        srf::enqueue_stream_sync_event(output->stream()).get();
        return output;
    }
}
