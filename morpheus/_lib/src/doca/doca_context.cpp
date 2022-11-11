#include "morpheus/doca/doca_context.hpp"
#include "morpheus/doca/common.h"
#include "morpheus/doca/samples/common.h"

#include <rte_eal.h>
#include <doca_version.h>
#include <doca_argp.h>
#include <doca_gpu.h>

#include <glog/logging.h>
#include <string>
#include <iostream>

namespace morpheus::doca
{

static doca_error_t get_dpdk_port_id_doca_dev(struct doca_dev *dev_input, uint16_t *port_id)
{
	struct doca_dev *dev_local = NULL;
	struct doca_pci_bdf pci_addr_local;
	struct doca_pci_bdf pci_addr_input;
	doca_error_t result;
	uint16_t dpdk_port_id;

	if (dev_input == NULL || port_id == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	*port_id = RTE_MAX_ETHPORTS;

	for (dpdk_port_id = 0; dpdk_port_id < RTE_MAX_ETHPORTS; dpdk_port_id++) {
		/* search for the probed devices */
		if (!rte_eth_dev_is_valid_port(dpdk_port_id))
			continue;

		result = doca_dpdk_port_as_dev(dpdk_port_id, &dev_local);
		if (result != DOCA_SUCCESS) {
      throw std::runtime_error("Failed to find DOCA device associated with port ID");
			// DOCA_LOG_ERR("Failed to find DOCA device associated with port ID %d: %s", dpdk_port_id, doca_get_error_string(result));
			// return result;
		}

		result = doca_devinfo_get_pci_addr(doca_dev_as_devinfo(dev_local), &pci_addr_local);
		if (result != DOCA_SUCCESS) {
      throw std::runtime_error("Failed to get device PCI address");
			// DOCA_LOG_ERR("Failed to get device PCI address: %s", doca_get_error_string(result));
			// return result;
		}

		result = doca_devinfo_get_pci_addr(doca_dev_as_devinfo(dev_input), &pci_addr_input);
		if (result != DOCA_SUCCESS) {
      throw std::runtime_error("Failed to get device PCI address");
			// DOCA_LOG_ERR("Failed to get device PCI address: %s", doca_get_error_string(result));
			// return result;
		}

		if (pci_addr_local.raw == pci_addr_input.raw) {
			*port_id = dpdk_port_id;
			break;
		}
	}

	// DOCA_LOG_INFO("dpdk port id %d", *port_id);
	return DOCA_SUCCESS;
}

doca_context::doca_context(std::string nic_addr, std::string gpu_addr):
  _max_queue_count(1)
{
    char* nic_addr_c = new char[nic_addr.size() + 1];
    char* gpu_addr_c = new char[gpu_addr.size() + 1];

    nic_addr_c[nic_addr.size()] = '\0';
    gpu_addr_c[gpu_addr.size()] = '\0';

    std::copy(nic_addr.begin(), nic_addr.end(), nic_addr_c);
    std::copy(gpu_addr.begin(), gpu_addr.end(), gpu_addr_c);

    auto argv = std::vector<char*>();

    char program[] = "";
    argv.push_back(program);

    char a_flag[] = "-a";
    // argv.push_back(a_flag);
    // argv.push_back(nic_addr_c);
    // argv.push_back(a_flag);
    // argv.push_back(gpu_addr_c);
    argv.push_back(a_flag);
    argv.push_back("00:00.0");

    // char l_flag[] = "-l";
    // char l_arg[] = "0,1,2,3,4";
    // argv.push_back(l_flag);
    // argv.push_back(l_arg);

    argv.push_back("--log-level");
    argv.push_back("eal,8");

    auto doca_ret_0 = parse_pci_addr(nic_addr_c, &_pci_bdf);
    if (doca_ret_0 != DOCA_SUCCESS) {
      throw std::runtime_error(
        "DOCA Failed to parse NIC device PCI address: " + std::string(doca_get_error_string(doca_ret_0))
      );
    }

	  auto doca_ret_1 = open_doca_device_with_pci(&_pci_bdf, nullptr, &_dev);
	  if (doca_ret_1 != DOCA_SUCCESS) {
      throw std::runtime_error(
        "DOCA Failed to open NIC device based on PCI address: " + std::string(doca_get_error_string(doca_ret_1))
      );
	  }

    auto doca_ret = doca_gpu_create(gpu_addr_c, &_gpu);
    if (doca_ret != DOCA_SUCCESS) {
      throw std::runtime_error(
        "DOCA initialization failed: " + std::string(doca_get_error_string(doca_ret))
      );
    }

    auto eal_ret = rte_eal_init(argv.size(), argv.data());
    if (eal_ret < 0) {
      throw std::runtime_error(
        "DPDK initialization failed: " + std::to_string(eal_ret)
      );
    }

    auto gpu_attack_dpdk_ret = doca_gpu_to_dpdk(_gpu);
    if (gpu_attack_dpdk_ret != DOCA_SUCCESS) {
      throw std::runtime_error(
        "DOCA to DPDK attach failed: " + std::string(doca_get_error_string(gpu_attack_dpdk_ret))
      );
    }

    auto ret_doca_2 = doca_dpdk_port_probe(_dev, "");
    if (ret_doca_2 != DOCA_SUCCESS) {
      throw std::runtime_error(
        "doca_dpdk_port_probe returned: " + std::string(doca_get_error_string(ret_doca_2)));
    }

    auto ret_doca_3 = get_dpdk_port_id_doca_dev(_dev, &_nic_port);
    if (ret_doca_3 != DOCA_SUCCESS) {
      throw std::runtime_error(
        "get_dpdk_port_id_doca_dev returned: " + std::string(doca_get_error_string(ret_doca_3)));
    }

    if (_nic_port == RTE_MAX_ETHPORTS) {
      throw std::runtime_error(
        "No DPDK port matches the DOCA device");
    }

    auto dpdk_config = [](){
      application_dpdk_config dpdk_config;
      dpdk_config.port_config.nb_ports = 1;
      dpdk_config.port_config.nb_queues = 1;
      dpdk_config.port_config.nb_hairpin_q = 0;
      dpdk_config.reserve_main_thread = true;
      return dpdk_config;
    }();

    _flow_port = init_doca_flow(_nic_port, _max_queue_count, &dpdk_config);
    if (_flow_port == nullptr) {
      throw std::runtime_error("DOCA Flow initialization failed");
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

doca_dev* doca_context::dev()
{
  return _dev;
}

uint16_t doca_context::nic_port()
{
  return _nic_port;
}

doca_flow_port* doca_context::flow_port()
{
  return _flow_port;
}

doca_rx_queue::doca_rx_queue(std::shared_ptr<doca_context> context):
  _context(context),
  _rxq_info_gpu(nullptr),
  _rxq_info_cpu(nullptr)
{
  auto ret = doca_gpu_rxq_create(
    _context->gpu(),
    _context->dev(),
    // desc_n,
    8192,
    // (app_cfg.receive_mode == RECEIVE_CPU ? DOCA_GPU_COMM_CPU : DOCA_GPU_COMM_GPU),
    DOCA_GPU_COMM_GPU,
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

doca_gpu_rxq_info* doca_rx_queue::rxq_info_cpu()
{
  return _rxq_info_cpu;
}

doca_gpu_rxq_info* doca_rx_queue::rxq_info_gpu()
{
  return _rxq_info_gpu;
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
