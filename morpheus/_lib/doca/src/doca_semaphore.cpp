/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "doca_semaphore.hpp"

namespace morpheus::doca {

DocaSemaphore::DocaSemaphore(std::shared_ptr<DocaContext> context, uint16_t size) :
  m_context(std::move(context)),
  m_size(size)
{
    DOCA_TRY(doca_gpu_semaphore_create(m_context->gpu(), &m_semaphore));
    DOCA_TRY(doca_gpu_semaphore_set_memory_type(m_semaphore, DOCA_GPU_MEM_CPU_GPU));
    DOCA_TRY(doca_gpu_semaphore_set_items_num(m_semaphore, size));
    DOCA_TRY(doca_gpu_semaphore_start(m_semaphore));
    DOCA_TRY(doca_gpu_semaphore_get_gpu_handle(m_semaphore, &m_semaphore_gpu));
}

DocaSemaphore::~DocaSemaphore()
{
    doca_gpu_semaphore_destroy(m_semaphore);
}

doca_gpu_semaphore_gpu* DocaSemaphore::gpu_ptr()
{
    return m_semaphore_gpu;
}

uint16_t DocaSemaphore::size()
{
    return m_size;
}

}  // namespace morpheus::doca
