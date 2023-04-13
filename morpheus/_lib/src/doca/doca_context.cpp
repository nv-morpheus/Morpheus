#define DOCA_ALLOW_EXPERIMENTAL_API

#include "morpheus/doca/doca_context.hpp"
#include "morpheus/utilities/error.hpp"
// #include "morpheus/doca/common.h"
// #include "morpheus/doca/samples/common.h"

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <doca_version.h>
#include <doca_argp.h>
#include <doca_dpdk.h>
#include <doca_gpunetio.h>
#include <doca_error.h>
#include <cuda_runtime.h>
#include <doca_eth_rxq.h>

#include <glog/logging.h>
#include <string>
#include <iostream>

// #define GPU_SUPPORT

namespace morpheus {

struct doca_error : public std::runtime_error {
  doca_error(std::string const& message) : std::runtime_error(message) {}
};

struct rte_error : public std::runtime_error {
  rte_error(std::string const& message) : std::runtime_error(message) {}
};

namespace detail {

inline void throw_doca_error(doca_error_t error, const char* file, unsigned int line)
{
  throw morpheus::doca_error(std::string{"DOCA error encountered at: " + std::string{file} + ":" +
                                     std::to_string(line) + ": " + std::to_string(error) + " " +
                                     std::string(doca_get_error_string(error))});
}

inline void throw_rte_error(int error, const char* file, unsigned int line)
{
  throw morpheus::rte_error(std::string{"RTE error encountered at: " + std::string{file} + ":" +
                                     std::to_string(line) + ": " + std::to_string(error)});
}

}

}

#define DOCA_TRY(call)                                                \
  do {                                                                \
    doca_error_t const status = (call);                               \
    if (DOCA_SUCCESS != status) {                                     \
      morpheus::detail::throw_doca_error(status, __FILE__, __LINE__); \
    }                                                                 \
  } while (0);

#define RTE_TRY(call)                                                \
  do {                                                                \
    int const status = (call);                               \
    if (status) {                                     \
      morpheus::detail::throw_rte_error(status, __FILE__, __LINE__); \
    }                                                                 \
  } while (0);

namespace {

static uint64_t default_flow_timeout_usec;

#define GPU_PAGE_SIZE (1UL << 16)
#define MAX_PKT_SIZE 8192
#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
#define MAX_PORT_STR_LEN 128 /* Maximal length of port name */
#define MAX_PKT_NUM 65536

/* Port configuration */
struct application_port_config {
	int nb_ports;				/* Set on init to 0 for don't care, required ports otherwise */
	int nb_queues;				/* Set on init to 0 for don't care, required minimum cores otherwise */
	int nb_hairpin_q;			/* Set on init to 0 to disable, hairpin queues otherwise */
	uint16_t enable_mbuf_metadata	:1;	/* Set on init to 0 to disable, otherwise it will add meta to each mbuf */
	uint16_t self_hairpin		:1;	/* Set on init to 1 enable both self and peer hairpin */
	uint16_t rss_support		:1;	/* Set on init to 0 for no RSS support, RSS support otherwise */
	uint16_t lpbk_support		:1;	/* Enable loopback support */
	uint16_t isolated_mode		:1;	/* Set on init to 0 for no isolation, isolated mode otherwise */
};

/* SFT configuration */
struct application_sft_config {
	bool enable;	  		/* Enable SFT */
	bool enable_ct;	  		/* Enable connection tracking feature of SFT */
	bool enable_frag; 		/* Enable fragmentation feature of SFT  */
	bool enable_state_hairpin;	/* Enable HW hairpin offload */
	bool enable_state_drop;		/* Enable HW drop offload */
};

struct gpu_pipeline {
	int gpu_id;					/* GPU id */
	bool gpu_support;				/* Enable program GPU support */
	bool is_host_mem;				/* Allocate mbuf's mempool on GPU or device memory */
	cudaStream_t c_stream;				/* CUDA stream */
	struct rte_pktmbuf_extmem ext_mem;		/* Mbuf's mempool */
	struct rte_gpu_comm_list *comm_list;		/* Communication list between device(GPU) and host(CPU) */
};

/* DPDK configuration */
struct application_dpdk_config {
	struct application_port_config port_config;	/* DPDK port configuration */
	struct application_sft_config sft_config;	/* DPDK SFT configuration */
	bool reserve_main_thread;			/* Reserve lcore for the main thread */
	struct rte_mempool *mbuf_pool;			/* Will be filled by "dpdk_queues_and_ports_init".
							 * Memory pool that will be used by the DPDK ports
							 * for allocating rte_pktmbuf
							 */
#ifdef GPU_SUPPORT
	struct gpu_pipeline pipe;			/* GPU pipeline */
#endif
};

doca_error_t parse_pci_addr(std::string const &addr, doca_pci_bdf &bdf)
{
	uint32_t tmpu;
	char tmps[4];

	if (addr.length() != 7 || addr[2] != ':' || addr[5] != '.')
		return DOCA_ERROR_INVALID_VALUE;

	tmps[0] = addr[0];
	tmps[1] = addr[1];
	tmps[2] = '\0';
	tmpu = strtoul(tmps, NULL, 16);
	if ((tmpu & 0xFFFFFF00) != 0)
		return DOCA_ERROR_INVALID_VALUE;
	bdf.bus = tmpu;

	tmps[0] = addr[3];
	tmps[1] = addr[4];
	tmps[2] = '\0';
	tmpu = strtoul(tmps, NULL, 16);
	if ((tmpu & 0xFFFFFFE0) != 0)
		return DOCA_ERROR_INVALID_VALUE;
	bdf.device = tmpu;

	tmps[0] = addr[6];
	tmps[1] = '\0';
	tmpu = strtoul(tmps, NULL, 16);
	if ((tmpu & 0xFFFFFFF8) != 0)
		return DOCA_ERROR_INVALID_VALUE;
	bdf.function = tmpu;

	return DOCA_SUCCESS;
}

typedef doca_error_t (*jobs_check)(struct doca_devinfo *);

doca_error_t
open_doca_device_with_pci(const struct doca_pci_bdf *value, jobs_check func, struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	struct doca_pci_bdf buf;
	size_t i;

	/* Set default return value */
	*retval = NULL;

	DOCA_TRY(doca_devinfo_list_create(&dev_list, &nb_devs));

	/* Search */
	for (i = 0; i < nb_devs; i++) {
    DOCA_TRY(doca_devinfo_get_pci_addr(dev_list[i], &buf));
		if (buf.raw == value->raw) {
			/* If any special capabilities are needed */
			if (func != NULL && func(dev_list[i]) != DOCA_SUCCESS)
				continue;

			/* if device can be opened */
			auto res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_list_destroy(dev_list);
				return res;
			}
		}
	}

	// DOCA_LOG_WARN("Matching device not found");

	doca_devinfo_list_destroy(dev_list);

	return DOCA_ERROR_NOT_FOUND;
}

struct doca_flow_port * init_doca_flow(uint16_t port_id, uint8_t rxq_num)
{
	char port_id_str[MAX_PORT_STR_LEN];
	struct doca_flow_port_cfg port_cfg = {0};
	struct doca_flow_port *df_port;
	struct doca_flow_cfg rxq_flow_cfg = {0};
	struct rte_eth_dev_info dev_info = {0};
	struct rte_eth_conf eth_conf = {
		.rxmode = {
			.mtu = 2048, /* Not really used, just to initialize DPDK */
		},
		.txmode = {
			.offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM,
		},
	};
	struct rte_mempool *mp = NULL;
	struct rte_eth_txconf tx_conf;
	struct rte_flow_error error;

	/*
	 * DPDK should be initialized and started before DOCA Flow.
	 * DPDK doesn't start the device without, at least, one DPDK Rx queue.
	 * DOCA Flow needs to specify in advance how many Rx queues will be used by the app.
	 *
	 * Following lines of code can be considered the minimum WAR for this issue.
	 */

  RTE_TRY(rte_eth_dev_info_get(port_id, &dev_info));
	RTE_TRY(rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf));

	mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, MAX_PKT_SIZE, rte_eth_dev_socket_id(port_id));
	if (mp == NULL) {
		// DOCA_LOG_ERR("Failed rte_pktmbuf_pool_create with: %s", rte_strerror(-ret));
    throw std::exception();
		// return NULL;
	}

	tx_conf = dev_info.default_txconf;
	tx_conf.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM;

	for (int idx = 0; idx < rxq_num; idx++) {
		RTE_TRY(rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), NULL, mp));
		RTE_TRY(rte_eth_tx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), &tx_conf));
	}

	RTE_TRY(rte_flow_isolate(port_id, 1, &error));
	RTE_TRY(rte_eth_dev_start(port_id));

	/* Initialize doca flow framework */
	rxq_flow_cfg.queues = rxq_num;
	/*
	 * HWS: Hardware steering
	 * Isolated: don't create RSS rule for DPDK created RX queues
	 */
	rxq_flow_cfg.mode_args = "vnf,hws,isolated";
	rxq_flow_cfg.resource.nb_counters = FLOW_NB_COUNTERS;

	DOCA_TRY(doca_flow_init(&rxq_flow_cfg));

	/* Start doca flow port */
	port_cfg.port_id = port_id;
	port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
	snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
	port_cfg.devargs = port_id_str;
	DOCA_TRY(doca_flow_port_start(&port_cfg, &df_port));

	default_flow_timeout_usec = 0;

	return df_port;
}

}

namespace morpheus::doca
{

static doca_error_t get_dpdk_port_id_doca_dev(struct doca_dev *dev_input, uint16_t *port_id)
{
	struct doca_dev *dev_local = NULL;
	struct doca_pci_bdf pci_addr_local;
	struct doca_pci_bdf pci_addr_input;
	uint16_t dpdk_port_id;

	if (dev_input == NULL || port_id == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	*port_id = RTE_MAX_ETHPORTS;

	for (dpdk_port_id = 0; dpdk_port_id < RTE_MAX_ETHPORTS; dpdk_port_id++) {
		/* search for the probed devices */
		if (!rte_eth_dev_is_valid_port(dpdk_port_id))
			continue;

		DOCA_TRY(doca_dpdk_port_as_dev(dpdk_port_id, &dev_local));
		DOCA_TRY(doca_devinfo_get_pci_addr(doca_dev_as_devinfo(dev_local), &pci_addr_local));
		DOCA_TRY(doca_devinfo_get_pci_addr(doca_dev_as_devinfo(dev_input), &pci_addr_input));

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

    RTE_TRY(rte_eal_init(argv.size(), argv.data()));

    DOCA_TRY(parse_pci_addr(nic_addr_c, _pci_bdf));
	  DOCA_TRY(open_doca_device_with_pci(&_pci_bdf, nullptr, &_dev));
    DOCA_TRY(doca_gpu_create(gpu_addr_c, &_gpu));


    // auto gpu_attack_dpdk_ret = doca_gpu_to_dpdk(_gpu);
    // if (gpu_attack_dpdk_ret != DOCA_SUCCESS) {
    //   throw std::runtime_error(
    //     "DOCA to DPDK attach failed: " + std::string(doca_get_error_string(gpu_attack_dpdk_ret))
    //   );
    // }

    DOCA_TRY(doca_dpdk_port_probe(_dev, ""));
    DOCA_TRY(get_dpdk_port_id_doca_dev(_dev, &_nic_port));
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

    _flow_port = init_doca_flow(_nic_port, _max_queue_count);
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

    auto doca_ret = doca_gpu_destroy(_gpu);
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
  _rxq_info_cpu(nullptr),
  _packet_buffer(nullptr),
  _doca_ctx(nullptr)
{
  auto ret = doca_eth_rxq_create(
    // _context->gpu(),
    // _context->dev(),
    // // desc_n,
    // 8192,
    // // (app_cfg.receive_mode == RECEIVE_CPU ? DOCA_GPU_COMM_CPU : DOCA_GPU_COMM_GPU),
    // DOCA_GPU_COMM_GPU,
    // // ((app_cfg.processing == PROCESSING_INFERENCE_HTTP) ? MAX_PKT_HTTP_PAYLOAD : MAX_PKT_PAYLOAD),
    // MAX_PKT_PAYLOAD,
    // // stride_num,
    // 65536,
    // DOCA_GPU_MEM_GPU,
    // false,
    // &_rxq_info_gpu,
    &_rxq_info_cpu
  );

  DOCA_TRY(doca_eth_rxq_set_num_packets(_rxq_info_cpu, MAX_PKT_NUM));
	DOCA_TRY(doca_eth_rxq_set_max_packet_size(_rxq_info_cpu, MAX_PKT_SIZE));

  uint32_t cyclic_buffer_size;

  DOCA_TRY(doca_eth_rxq_get_pkt_buffer_size(_rxq_info_cpu, &cyclic_buffer_size));
  DOCA_TRY(doca_mmap_create(nullptr, &_packet_buffer));
  DOCA_TRY(doca_mmap_dev_add(_packet_buffer, context->dev()));

  void* packet_address;

  DOCA_TRY(doca_gpu_mem_alloc(
    context->gpu(),
    cyclic_buffer_size,
    GPU_PAGE_SIZE,
    DOCA_GPU_MEM_GPU,
    &packet_address,
    nullptr
  ));

  DOCA_TRY(doca_mmap_set_memrange(_packet_buffer, packet_address, cyclic_buffer_size));
  DOCA_TRY(doca_mmap_set_permissions(_packet_buffer, DOCA_ACCESS_LOCAL_READ_WRITE));
  DOCA_TRY(doca_mmap_start(_packet_buffer));
  DOCA_TRY(doca_eth_rxq_set_pkt_buffer(_rxq_info_cpu, _packet_buffer, 0, cyclic_buffer_size));

  _doca_ctx = doca_eth_rxq_as_doca_ctx(_rxq_info_cpu);

  if (_doca_ctx == nullptr){
    MORPHEUS_FAIL("unable to rxq as doca ctx ?");
  }

  DOCA_TRY(doca_ctx_dev_add(_doca_ctx, context->dev()));
  DOCA_TRY(doca_ctx_set_datapath_on_gpu(_doca_ctx, context->gpu()));
  DOCA_TRY(doca_ctx_start(_doca_ctx));
  DOCA_TRY(doca_eth_rxq_get_gpu_handle(_rxq_info_cpu, &_rxq_info_gpu));
}

doca_rx_queue::~doca_rx_queue()
{
  // DOCA_TRY(doca_gpu_rxq_destroy(_rxq_info_cpu));
}

doca_eth_rxq* doca_rx_queue::rxq_info_cpu()
{
  return _rxq_info_cpu;
}

doca_gpu_eth_rxq* doca_rx_queue::rxq_info_gpu()
{
  return _rxq_info_gpu;
}

doca_rx_pipe::doca_rx_pipe(
  std::shared_ptr<doca_context> context,
  std::shared_ptr<doca_rx_queue> rxq,
  uint32_t source_ip_filter
):
    _context(context),
    _rxq(rxq),
    _pipe(nullptr)
{
  uint16_t flow_queue_id;
  uint16_t rss_queues[1];
  doca_eth_rxq_get_flow_queue_id(rxq->rxq_info_cpu(), &flow_queue_id);
  rss_queues[0] = flow_queue_id;

	struct doca_flow_match match_mask = {0};
	struct doca_flow_match match = ([](){
    doca_flow_match match;
    match.outer = ([](){
      doca_flow_header_format outer;
			outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
			outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
      return outer;
    })();
    return match;
	})();

  auto fwd = ([&](){
    doca_flow_fwd fwd;
		fwd.type = DOCA_FLOW_FWD_RSS;
		fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP;
		fwd.rss_queues = rss_queues;
		fwd.num_of_queues = 1;
    return fwd;
	})();

	auto miss_fwd = ([](){
    doca_flow_fwd miss_fwd;
		miss_fwd.type = DOCA_FLOW_FWD_DROP;
    return miss_fwd;
	})();

	auto monitor = ([](){
    doca_flow_monitor monitor;
		monitor.flags = DOCA_FLOW_MONITOR_COUNT;
    return monitor;
	})();

	auto pipe_cfg = ([&](){
    doca_flow_pipe_cfg pipe_cfg;
		pipe_cfg.attr = ([](){
      doca_flow_pipe_attr attr;
			attr.name = "GPU_RXQ_TCP_PIPE";
			attr.type = DOCA_FLOW_PIPE_BASIC;
			attr.nb_actions = 0;
			attr.is_root = false;
      return attr;
		})(),
		pipe_cfg.match = &match;
		pipe_cfg.match_mask = &match_mask;
		pipe_cfg.monitor = &monitor;
		pipe_cfg.port = context->flow_port();
    return pipe_cfg;
	})();

  DOCA_TRY(doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &_pipe));
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
  DOCA_TRY(doca_gpu_semaphore_create(_context->gpu(), &_semaphore));
	DOCA_TRY(doca_gpu_semaphore_set_memory_type(_semaphore, DOCA_GPU_MEM_CPU_GPU));
  DOCA_TRY(doca_gpu_semaphore_set_items_num(_semaphore, size));
  DOCA_TRY(doca_gpu_semaphore_start(_semaphore));
  DOCA_TRY(doca_gpu_semaphore_get_gpu_handle(_semaphore, &_semaphore_gpu));
}

doca_semaphore::~doca_semaphore()
{
  // can't destroy semaphore... ?
}


doca_gpu_semaphore_gpu* doca_semaphore::in_gpu()
{
  return _semaphore_gpu;
}

uint16_t doca_semaphore::size()
{
  return _size;
}

}
