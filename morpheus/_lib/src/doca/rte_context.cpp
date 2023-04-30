#include "morpheus/doca/rte_context.hpp"

#include "morpheus/doca/error.hpp"

#include <glog/logging.h>
#include <rte_eal.h>

#include <array>

namespace morpheus::doca {

RTEContext::RTEContext()
{
    auto argv = std::vector<char*>();

    std::array<char, 1> fake_program = {""};
    argv.push_back(fake_program.begin());

    std::array<char, 3> a_flag           = {"-a"};
    std::array<char, 8> fake_pci_address = {"00:00.0"};
    argv.push_back(a_flag.begin());
    argv.push_back(fake_pci_address.begin());

    // argv.push_back("-l");
    // argv.push_back("0,1,2,3,4");

    // argv.push_back("--log-level");
    // argv.push_back("eal,8");

    RTE_TRY(rte_eal_init(argv.size(), argv.data()));
}

RTEContext::~RTEContext()
{
    auto eal_ret = rte_eal_cleanup();

    if (eal_ret < 0)
    {
        LOG(WARNING) << "EAL cleanup failed (" << eal_ret << ")" << std::endl;
    }
}

}  // namespace morpheus::doca