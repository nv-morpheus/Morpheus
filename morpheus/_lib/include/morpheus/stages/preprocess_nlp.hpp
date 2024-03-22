/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/utilities/matx_util.hpp"

#include <boost/fiber/context.hpp>
#include <boost/fiber/future/future.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <mrc/node/rx_sink_base.hpp>
#include <mrc/node/rx_source_base.hpp>
#include <mrc/node/sink_properties.hpp>
#include <mrc/node/source_properties.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/types.hpp>
#include <nvtext/normalize.hpp>
#include <nvtext/subword_tokenize.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, from
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

#include <cstdint>  // for uint32_t
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** PreprocessNLPStage**********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief NLP input data for inference
 */
template <typename InputT, typename OutputT>
class PreprocessNLPStage : public mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Preprocess NLP Stage object
     *
     * @param vocab_hash_file : Path to hash file containing vocabulary of words with token-ids. This can be created
     * from the raw vocabulary using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
     * @param sequence_length : Sequence Length to use (We add to special tokens for NER classification job).
     * @param truncation : If set to true, strings will be truncated and padded to max_length. Each input string will
     * result in exactly one output sequence. If set to false, there may be multiple output sequences when the
     * max_length is smaller than generated tokens.
     * @param do_lower_case : If set to true, original text will be lowercased before encoding.
     * @param add_special_token : Whether or not to encode the sequences with the special tokens of the BERT
     * classification model.
     * @param stride : If `truncation` == False and the tokenized string is larger than max_length, the sequences
     * containing the overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is
     * equal to stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will
     * be repeated on the second sequence and so on until the entire sentence is encoded.
     * @param column : Name of the string column to operate on, defaults to "data".
     */
    PreprocessNLPStage(std::string vocab_hash_file,
                       uint32_t sequence_length,
                       bool truncation,
                       bool do_lower_case,
                       bool add_special_token,
                       int stride         = -1,
                       std::string column = "data");

  private:
    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    std::string m_vocab_hash_file;
    std::string m_column;
    uint32_t m_sequence_length;
    bool m_truncation;
    bool m_do_lower_case;
    bool m_add_special_token;
    int m_stride{-1};
};

template <typename InputT, typename OutputT>
PreprocessNLPStage<InputT, OutputT>::PreprocessNLPStage(std::string vocab_hash_file,
                                                        uint32_t sequence_length,
                                                        bool truncation,
                                                        bool do_lower_case,
                                                        bool add_special_token,
                                                        int stride,
                                                        std::string column) :
  base_t(base_t::op_factory_from_sub_fn(build_operator())),
  m_vocab_hash_file(std::move(vocab_hash_file)),
  m_sequence_length(sequence_length),
  m_truncation(truncation),
  m_do_lower_case(do_lower_case),
  m_add_special_token(add_special_token),
  m_stride(stride),
  m_column(std::move(column))
{}

nvtext::tokenizer_result subword_tokenize(const std::string& vocab_hash_file,
                                          uint32_t sequence_length,
                                          bool do_lower_case,
                                          bool truncation,
                                          cudf::strings_column_view const& string_col,
                                          int stride,
                                          rmm::mr::device_memory_resource* mr);

template <typename InputT, typename OutputT>
PreprocessNLPStage<InputT, OutputT>::subscribe_fn_t PreprocessNLPStage<InputT, OutputT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        uint32_t stride = m_stride;

        // Auto calc stride to be 75% of sequence length
        if (stride < 0)
        {
            stride = m_sequence_length / 2;
            stride = stride + stride / 2;
        }

        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output, stride](sink_type_t x) {
                if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiMessage>>)
                {
                    // Convert to string view
                    auto meta = x->get_meta(this->m_column);

                    auto col        = meta.get_column(0);
                    auto string_col = cudf::strings_column_view{col};

                    auto token_results = subword_tokenize(this->m_vocab_hash_file,
                                                          this->m_sequence_length,
                                                          this->m_do_lower_case,
                                                          this->m_truncation,
                                                          string_col,
                                                          stride,
                                                          rmm::mr::get_current_device_resource());

                    // Build the results
                    auto memory = std::make_shared<InferenceMemory>(token_results.nrows_tensor);

                    TensorIndex length = token_results.tensor_token_ids->size() / token_results.sequence_length;
                    auto input_ids_released =
                        cudf::cast(token_results.tensor_token_ids->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();

                    memory->set_tensor("input_ids",
                                       Tensor::create(std::move(input_ids_released.data),
                                                      DType::create<int32_t>(),
                                                      {length, static_cast<TensorIndex>(token_results.sequence_length)},
                                                      {},
                                                      0));

                    length = token_results.tensor_attention_mask->size() / token_results.sequence_length;
                    auto input_mask_released =
                        cudf::cast(token_results.tensor_attention_mask->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();
                    memory->set_tensor("input_mask",
                                       Tensor::create(std::move(input_mask_released.data),
                                                      DType::create<int32_t>(),
                                                      {length, static_cast<TensorIndex>(token_results.sequence_length)},
                                                      {},
                                                      0));

                    auto tensor_index_dtype = DType::create<TensorIndex>();
                    length                  = token_results.tensor_metadata->size() / 3;
                    auto seq_ids_released   = cudf::cast(token_results.tensor_metadata->view(),
                                                       cudf::data_type(tensor_index_dtype.cudf_type_id()))
                                                ->release();

                    std::shared_ptr<rmm::device_buffer> seq_ids_data = std::move(seq_ids_released.data);

                    if (x->mess_offset > 0)
                    {
                        // Add an offset to the seq_ids so the message IDs line up
                        MatxUtil::offset_seq_ids(
                            DevMemInfo{seq_ids_data, tensor_index_dtype.type_id(), {length, 3}, {1, 3}},
                            x->mess_offset);
                    }

                    memory->set_tensor("seq_ids", Tensor::create(seq_ids_data, tensor_index_dtype, {length, 3}, {}, 0));

                    auto next = std::make_shared<MultiInferenceMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                    output.on_next(std::move(next));
                }
                else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
                {
                    // Convert to string view
                    auto num_columns = x->payload()->get_info().num_columns();
                    auto meta =
                        x->payload()->get_info().get_slice(0, num_columns, std::vector<std::string>{this->m_column});

                    auto col        = meta.get_column(0);
                    auto string_col = cudf::strings_column_view{col};

                    auto token_results = subword_tokenize(this->m_vocab_hash_file,
                                                          this->m_sequence_length,
                                                          this->m_do_lower_case,
                                                          this->m_truncation,
                                                          string_col,
                                                          stride,
                                                          rmm::mr::get_current_device_resource());

                    // Build the results
                    auto memory = std::make_shared<TensorMemory>(token_results.nrows_tensor);

                    TensorIndex length = token_results.tensor_token_ids->size() / token_results.sequence_length;
                    auto input_ids_released =
                        cudf::cast(token_results.tensor_token_ids->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();

                    memory->set_tensor("input_ids",
                                       Tensor::create(std::move(input_ids_released.data),
                                                      DType::create<int32_t>(),
                                                      {length, static_cast<TensorIndex>(token_results.sequence_length)},
                                                      {},
                                                      0));

                    length = token_results.tensor_attention_mask->size() / token_results.sequence_length;
                    auto input_mask_released =
                        cudf::cast(token_results.tensor_attention_mask->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();
                    memory->set_tensor("input_mask",
                                       Tensor::create(std::move(input_mask_released.data),
                                                      DType::create<int32_t>(),
                                                      {length, static_cast<TensorIndex>(token_results.sequence_length)},
                                                      {},
                                                      0));

                    auto tensor_index_dtype = DType::create<TensorIndex>();
                    length                  = token_results.tensor_metadata->size() / 3;
                    auto seq_ids_released   = cudf::cast(token_results.tensor_metadata->view(),
                                                       cudf::data_type(tensor_index_dtype.cudf_type_id()))
                                                ->release();

                    std::shared_ptr<rmm::device_buffer> seq_ids_data = std::move(seq_ids_released.data);

                    memory->set_tensor("seq_ids", Tensor::create(seq_ids_data, tensor_index_dtype, {length, 3}, {}, 0));

                    auto next = x;
                    next->tensors(memory);

                    output.on_next(std::move(next));
                }
                // sink_type_t not supported
                else
                {
                    std::string error_msg{"PreProcessNLPStage receives unsupported input type: " + std::string(typeid(x).name())};
                    LOG(ERROR) << error_msg;
                    throw std::runtime_error(error_msg);
                }
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

/****** PreprocessNLPStageInferenceProxy********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct PreprocessNLPStageInterfaceProxy
{
    /**
     * @brief Create and initialize a ProcessNLPStage that receives MultiMessage and emits MultiInferenceMessage, and
     * return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param vocab_hash_file : Path to hash file containing vocabulary of words with token-ids. This can be created
     * from the raw vocabulary using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
     * @param sequence_length : Sequence Length to use (We add to special tokens for NER classification job).
     * @param truncation : If set to true, strings will be truncated and padded to max_length. Each input string will
     * result in exactly one output sequence. If set to false, there may be multiple output sequences when the
     * max_length is smaller than generated tokens.
     * @param do_lower_case : If set to true, original text will be lowercased before encoding.
     * @param add_special_token : Whether or not to encode the sequences with the special tokens of the BERT
     * classification model.
     * @param stride : If `truncation` == False and the tokenized string is larger than max_length, the sequences
     * containing the overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is
     * equal to stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will
     * be repeated on the second sequence and so on until the entire sentence is encoded.
     * @param column : Name of the string column to operate on, defaults to "data".
     * @return std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>>
     */
    static std::shared_ptr<mrc::segment::Object<PreprocessNLPStage<MultiMessage, MultiInferenceMessage>>> init_multi(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::string vocab_hash_file,
        uint32_t sequence_length,
        bool truncation,
        bool do_lower_case,
        bool add_special_token,
        int stride         = -1,
        std::string column = "data");
    /**
     * @brief Create and initialize a ProcessNLPStage that receives ControlMessage and emits ControlMessage, and return
     * the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param vocab_hash_file : Path to hash file containing vocabulary of words with token-ids. This can be created
     * from the raw vocabulary using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
     * @param sequence_length : Sequence Length to use (We add to special tokens for NER classification job).
     * @param truncation : If set to true, strings will be truncated and padded to max_length. Each input string will
     * result in exactly one output sequence. If set to false, there may be multiple output sequences when the
     * max_length is smaller than generated tokens.
     * @param do_lower_case : If set to true, original text will be lowercased before encoding.
     * @param add_special_token : Whether or not to encode the sequences with the special tokens of the BERT
     * classification model.
     * @param stride : If `truncation` == False and the tokenized string is larger than max_length, the sequences
     * containing the overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is
     * equal to stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will
     * be repeated on the second sequence and so on until the entire sentence is encoded.
     * @param column : Name of the string column to operate on, defaults to "data".
     * @return std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>>
     */
    static std::shared_ptr<mrc::segment::Object<PreprocessNLPStage<ControlMessage, ControlMessage>>> init_cm(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::string vocab_hash_file,
        uint32_t sequence_length,
        bool truncation,
        bool do_lower_case,
        bool add_special_token,
        int stride         = -1,
        std::string column = "data");
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
