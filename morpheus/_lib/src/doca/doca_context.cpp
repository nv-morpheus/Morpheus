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

static uint32_t *cpu_exit_condition;
static uint32_t *gpu_exit_condition;

doca_context::doca_context(std::string nic_addr, std::string gpu_addr)
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
    argv.push_back(nic_addr_c);
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

    uint8_t app_cfg_nic_port = 0;
    uint8_t app_cfg_queue_num = 1;
    auto df_port = init_doca_flow(app_cfg_nic_port, app_cfg_queue_num, &dpdk_config);
    if (df_port == nullptr) {
      throw std::runtime_error("DOCA Flow initialization failed");
    }

    auto gpu_attack_dpdk_ret = doca_gpu_to_dpdk(_gpu);
    if (gpu_attack_dpdk_ret != DOCA_SUCCESS) {
      throw std::runtime_error("DOCA to DPDK attach failed, error=" + std::to_string(gpu_attack_dpdk_ret));
    }

    auto ret = doca_gpu_mem_alloc(_gpu, sizeof(uint32_t), MEM_ALIGN_SZ, DOCA_GPU_MEM_CPU_GPU, (void **)&gpu_exit_condition, (void **)&cpu_exit_condition);
    if (ret != DOCA_SUCCESS) {
      throw std::runtime_error("DOCA memalloc failed, error=" + std::to_string(gpu_attack_dpdk_ret));
    }
    memset(cpu_exit_condition, 0, app_cfg_queue_num * sizeof(uint32_t));

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

}
