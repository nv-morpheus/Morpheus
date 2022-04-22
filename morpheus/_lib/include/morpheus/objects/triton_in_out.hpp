//
// Created by drobison on 3/14/22.
//

#pragma once

#include <morpheus/utilities/type_util.hpp>

#include <string>
#include <vector>

namespace morpheus {
    /****** Component public implementations *******************/
    /****** TritonInOut****************************************/
    /**
     * TODO(Documentation)
     */
    struct TritonInOut {
        std::string name;
        size_t bytes;
        DType datatype;
        std::vector<int> shape;
        std::string mapped_name;
        size_t offset;
    };
}
