#include <glog/logging.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <iterator>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#pragma once

using namespace std::chrono_literals;

using time_point_t = std::chrono::time_point<std::chrono::system_clock>;

struct LLMClientRequest
{
    LLMClientRequest() = delete;
    LLMClientRequest(std::vector<std::string>&& requests,
                     std::promise<std::vector<std::string>>&& promise,
                     time_point_t requestsed_at) :
      m_queries(std::move(requests)),
      m_promise(std::move(promise)),
      m_requested_at(requestsed_at)
    {}

    std::vector<std::string> m_queries;
    std::promise<std::vector<std::string>> m_promise;
    time_point_t m_requested_at;
};

class LLMClientBatcher
{
  public:
    LLMClientBatcher()
    {
        m_batching_thread = std::thread(&LLMClientBatcher::batch_processing_loop, this);
    }

    ~LLMClientBatcher()
    {
        if (m_batching_thread.joinable())
        {
            m_loop_ct = true;
            m_batching_thread.join();
        }
    }

    std::future<std::vector<std::string>> generate(std::vector<std::string> requests)
    {
        std::promise<std::vector<std::string>> responses_promise{};

        auto responses_future = responses_promise.get_future();

        {
            auto guard = std::lock_guard(this->m_batches_mutex);
            m_requests.insert(m_requests.end(),
                              {
                                  std::move(requests),
                                  std::move(responses_promise),
                                  std::chrono::system_clock::now()  //
                              });
        }

        return responses_future;
    }

  private:
    void batch_processing_loop()
    {
        while (not m_loop_ct)
        {
            std::this_thread::sleep_for(m_loop_wait);

            auto guard = std::lock_guard(this->m_batches_mutex);

            if (m_requests.size() == 0)
            {
                break;
            }

            auto batch_size = 0;

            for (auto& request : m_requests)
            {
                batch_size += request.m_queries.size();
            }

            auto oldest_request_wait = std::chrono::system_clock::now() - m_requests.begin()->m_requested_at;

            // only go to next step if our batch size is big enough, our wait time is too long, or we're exiting.
            if (batch_size < m_min_batch_size and oldest_request_wait < m_max_wait and not m_loop_ct)
            {
                break;
            }

            auto batch = std::move(m_requests);
            m_requests = {};

            LOG(INFO) << "LLMClientBatcher: processing " << batch.size() << " batches with " << batch_size
                      << " total requests";
        }

        try
        {
            throw std::runtime_error("batcher destroyed");
        } catch (...)
        {
            for (auto& request : m_requests)
            {
                request.m_promise.set_exception(std::exception_ptr());
            }
        }
    }

    std::vector<LLMClientRequest> m_requests{};
    std::thread m_batching_thread;
    std::mutex m_batches_mutex{};
    uint32_t m_min_batch_size                 = 10;
    std::chrono::duration<long> m_max_wait    = 1s;
    std::chrono::duration<double> m_loop_wait = 0.001s;
    std::atomic<bool> m_loop_ct               = false;
};
