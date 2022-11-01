#pragma once

#include <doca_gpu.h>
#include <doca_flow.h>

#include <string>
#include <memory>

namespace morpheus::doca
{

#pragma GCC visibility push(default)

struct doca_context
{
private:
  doca_gpu* _gpu;
  doca_flow_port* _flow_port;
  uint32_t _nic_port;
  uint32_t _max_queue_count;

public:
  doca_context(std::string nic_addr, std::string gpu_addr);
  ~doca_context();

  doca_gpu* gpu();
  uint32_t nic_port();
  doca_flow_port* flow_port();
};

struct doca_rx_queue
{
  private:
    std::shared_ptr<doca_context> _context;
    doca_gpu_rxq_info* _rxq_info_gpu;
    doca_gpu_rxq_info* _rxq_info_cpu;

  public:
    doca_rx_queue(std::shared_ptr<doca_context> context);
    ~doca_rx_queue();

    doca_gpu_rxq_info* rxq_info_cpu();
    doca_gpu_rxq_info* rxq_info_gpu();
};

struct doca_rx_pipe
{
  private:
    std::shared_ptr<doca_context> _context;
    std::shared_ptr<doca_rx_queue> _rxq;
    doca_flow_pipe* _pipe;

  public:
    doca_rx_pipe(
      std::shared_ptr<doca_context> context,
      std::shared_ptr<doca_rx_queue> rxq
    );
    ~doca_rx_pipe();
};

struct doca_semaphore
{
  private:
    std::shared_ptr<doca_context> _context;
    uint16_t _size;
    doca_gpu_semaphore* _semaphore;
    doca_gpu_semaphore_in* _semaphore_in_gpu;
    doca_gpu_semaphore_in* _semaphore_in_cpu;
    doca_gpu_semaphore_in* _semaphore_info_gpu;
    doca_gpu_semaphore_in* _semaphore_info_cpu;

  public:
    doca_semaphore(std::shared_ptr<doca_context> context, uint16_t size);
    ~doca_semaphore();

    doca_gpu_semaphore_in* in_gpu();
    uint16_t size();
};

// template<typename T>
// struct doca_memmap_scalar
// {
//   private:
//    T* gpu_ptr;
//    T* cpu_ptr;

//   public:
//     doca_memmap_scalar(T initial_value);
//     ~doca_memmap_scalar();
// };

#pragma GCC visibility pop

}
