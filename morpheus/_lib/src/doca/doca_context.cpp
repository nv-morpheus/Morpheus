#include "morpheus/doca/doca_context.hpp"
#include "morpheus/doca/common.h"

#include <rte_eal.h>
#include <doca_version.h>
#include <doca_argp.h>
#include <doca_gpu.h>

#include <glog/logging.h>
#include <string>

namespace morpheus::doca
{

doca_context::doca_context(std::string nic_addr, std::string gpu_addr):
  _nic_port(0),
  _max_queue_count(4)
{
    char* nic_addr_c = new char[nic_addr.size() + 1];
    char* gpu_addr_c = new char[gpu_addr.size() + 1];

    nic_addr_c[nic_addr.size()] = '\0';
    gpu_addr_c[gpu_addr.size()] = '\0';

    std::copy(nic_addr.begin(), nic_addr.end(), nic_addr_c);
    std::copy(gpu_addr.begin(), gpu_addr.end(), gpu_addr_c);

    auto argv = std::vector<char*>();

    char program[] = "not a real program path";
    argv.push_back(program);

    char a_flag[] = "-a";
    argv.push_back(a_flag);
    argv.push_back(gpu_addr_c);

    char l_flag[] = "-l";
    char l_arg[] = "0,1,2,3,4";
    argv.push_back(l_flag);
    argv.push_back(l_arg);

    auto eal_ret = rte_eal_init(argv.size(), argv.data());
    if (eal_ret < 0) {
      throw std::runtime_error("EAL initialization failed, error=" + std::to_string(eal_ret));
    }

    auto doca_ret = doca_gpu_create(gpu_addr_c, &_gpu);
    if (doca_ret != DOCA_SUCCESS) {
      throw std::runtime_error("DOCA initialization failed, error=" + std::to_string(doca_ret));
    }

    auto dpdk_config = [](){
      application_dpdk_config dpdk_config;
      dpdk_config.port_config.nb_ports = 1;
      dpdk_config.port_config.nb_queues = 0;
      dpdk_config.port_config.nb_hairpin_q = 0;
      dpdk_config.reserve_main_thread = true;
      return dpdk_config;
    }();

    // uint8_t app_cfg_nic_port = 0;
    // uint8_t app_cfg_queue_num = 1;
    _flow_port = init_doca_flow(_nic_port, _max_queue_count, &dpdk_config);
    if (_flow_port == nullptr) {
      throw std::runtime_error("DOCA Flow initialization failed");
    }

    auto gpu_attack_dpdk_ret = doca_gpu_to_dpdk(_gpu);
    if (gpu_attack_dpdk_ret != DOCA_SUCCESS) {
      throw std::runtime_error("DOCA to DPDK attach failed, error=" + std::to_string(gpu_attack_dpdk_ret));
    }

    delete[] nic_addr_c;
    delete[] gpu_addr_c;
}

doca_context::~doca_context()
{
    auto eal_ret = rte_eal_cleanup();
    if (eal_ret < 0) {
      LOG(WARNING) << "EAL cleanup failed (" << eal_ret << ")" << std::endl;
    }

    auto doca_ret = doca_gpu_destroy(&_gpu);
    if (doca_ret != DOCA_SUCCESS) {
      LOG(WARNING) << "DOCA cleanup failed (" << doca_ret << ")" << std::endl;
    }
}

doca_gpu* doca_context::gpu()
{
  return _gpu;
}

uint32_t doca_context::nic_port()
{
  return _nic_port;
}

doca_flow_port* doca_context::flow_port()
{
  return _flow_port;
}

doca_gpu_rxq_info* doca_rx_queue::rxq_info_cpu()
{
  return _rxq_info_cpu;
}

doca_gpu_rxq_info* doca_rx_queue::rxq_info_gpu()
{
  return _rxq_info_gpu;
}

doca_rx_queue::doca_rx_queue(std::shared_ptr<doca_context> context):
  _context(context),
  _rxq_info_gpu(nullptr),
  _rxq_info_cpu(nullptr)
{
  auto ret = doca_gpu_rxq_create(
    _context->gpu(),
    _context->nic_port(),
    // desc_n,
    8192,
    // (app_cfg.receive_mode == RECEIVE_CPU ? DOCA_GPU_COMM_CPU : DOCA_GPU_COMM_GPU),
    DOCA_GPU_COMM_CPU,
    // ((app_cfg.processing == PROCESSING_INFERENCE_HTTP) ? MAX_PKT_HTTP_PAYLOAD : MAX_PKT_PAYLOAD),
    MAX_PKT_PAYLOAD,
    // stride_num,
    65536,
    DOCA_GPU_MEM_GPU,
    false,
    &_rxq_info_gpu,
    &_rxq_info_cpu
  );

  if (ret != DOCA_SUCCESS) {
    throw std::runtime_error("doca_gpu_rxq_create returned " + std::to_string(ret));
  }
}

doca_rx_queue::~doca_rx_queue()
{
  auto ret = doca_gpu_rxq_destroy(_rxq_info_cpu);
  if (ret != DOCA_SUCCESS) {
    LOG(WARNING) << "doca_gpu_rxq_destroy returned " << ret;
  }
}

doca_rx_pipe::doca_rx_pipe(
  std::shared_ptr<doca_context> context,
  std::shared_ptr<doca_rx_queue> rxq
):
    _context(context),
    _rxq(rxq)
{
  // TODO: inline this function and adjust accordingly. That, or updte build_rxq_pipe to accept IP Address / packet type
  _pipe = build_rxq_pipe(
    _context->nic_port(),
    _context->flow_port(),
    // rxq id,
    0,
    _rxq->rxq_info_cpu()->dpdk_idx,
    // is_tcp
    true
  );

  if(_pipe == nullptr) {
    throw std::runtime_error("build_rxq_pipe failed");
  }
}

doca_rx_pipe::~doca_rx_pipe()
{
  // can't destroy pipe... ?
  // doca_flow_destroy_pipe(_context->nic_port(), _pipe);
}

doca_semaphore::doca_semaphore(
  std::shared_ptr<doca_context> context,
  uint16_t size
):
  _context(context),
  _size(size)
{
  auto create_ret = doca_gpu_semaphore_create(
    _context->gpu(),
    size,
    DOCA_GPU_MEM_GPU_CPU,
    sizeof(proxy_sem_out),
    DOCA_GPU_MEM_CPU_GPU,
    &_semaphore
  );

  if (create_ret != DOCA_SUCCESS) {
    throw std::runtime_error("fail to create semaphore: " + std::to_string(create_ret));
    // DOCA_LOG_ERR("doca_gpu_semaphore_create returned %d\n", ret);
    // goto exit;
  }

  auto get_in_ret = doca_gpu_semaphore_get_in(
    _semaphore,
    &_semaphore_in_gpu,
    &_semaphore_in_cpu
  );

  if (get_in_ret != DOCA_SUCCESS) {
    throw std::runtime_error("fail to get semaphore in: " + std::to_string(get_in_ret));
  //   DOCA_LOG_ERR("doca_gpu_semaphore_get_in returned %d\n", ret);
  //   goto exit;
  }

  auto get_info_ret = doca_gpu_semaphore_get_info(
    _semaphore,
    (void **)&_semaphore_info_gpu,
    (void **)&_semaphore_info_cpu
  );

  if (get_info_ret != DOCA_SUCCESS) {
    throw std::runtime_error("fail to get semaphore info: " + std::to_string(get_info_ret));
    // DOCA_LOG_ERR("doca_gpu_semaphore_get_info returned %d\n", ret);
    // goto exit;
  }
}

doca_semaphore::~doca_semaphore()
{
  // can't destroy semaphore... ?
}


doca_gpu_semaphore_in* doca_semaphore::in_gpu()
{
  return _semaphore_in_gpu;
}

uint16_t doca_semaphore::size()
{
  return _size;
}

}
