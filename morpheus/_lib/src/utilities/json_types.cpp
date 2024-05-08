
#include "morpheus/utilities/json_types.hpp"

namespace {
template <typename T>
uint64_t type_to_uint64()
{
    return std::hash<std::string>{}(typeid(T).name());
}
}  // namespace

namespace morpheus::utilities {

PythonByteContainer::PythonByteContainer(mrc::pymrc::PyHolder py_obj) : m_py_obj(std::move(py_obj)) {}

mrc::pymrc::PyHolder PythonByteContainer::get_py_obj() const
{
    return m_py_obj;
}

py::object cast_from_json(const morpheus::utilities::json_t& source)
{
    if (source.is_null())
    {
        return py::none();
    }
    if (source.is_array())
    {
        py::list list_;
        for (const auto& element : source)
        {
            list_.append(cast_from_json(element));
        }
        return std::move(list_);
    }
    if (source.is_boolean())
    {
        return py::bool_(source.get<bool>());
    }
    if (source.is_number_float())
    {
        return py::float_(source.get<double>());
    }
    if (source.is_number_integer())
    {
        return py::int_(source.get<morpheus::utilities::json_t::number_integer_t>());
    }
    if (source.is_number_unsigned())
    {
        return py::int_(source.get<morpheus::utilities::json_t::number_unsigned_t>());
    }
    if (source.is_object())
    {
        py::dict dict;
        for (const auto& it : source.items())
        {
            dict[py::str(it.key())] = cast_from_json(it.value());
        }

        return std::move(dict);
    }
    if (source.is_string())
    {
        return py::str(source.get<std::string>());
    }
    if (source.is_binary())
    {
        if (source.get_binary().has_subtype() && source.get_binary().subtype() == type_to_uint64<py::object>())
        {
            return source.get_binary().get_py_obj();
        }
        throw std::runtime_error("Unsupported binary type");
    }

    return py::none();
}

json_t cast_from_pyobject_impl(const py::object& source,
                               mrc::pymrc::unserializable_handler_fn_t unserializable_handler_fn,
                               const std::string& parent_path = "")
{
    // Dont return via initializer list with JSON. It performs type deduction and gives different results
    // NOLINTBEGIN(modernize-return-braced-init-list)
    if (source.is_none())
    {
        return json_t();
    }

    if (py::isinstance<py::dict>(source))
    {
        const auto py_dict = source.cast<py::dict>();
        auto json_obj      = json_t::object();
        for (const auto& p : py_dict)
        {
            std::string key{p.first.cast<std::string>()};
            std::string path{parent_path + "/" + key};
            json_obj[key] = cast_from_pyobject_impl(p.second.cast<py::object>(), unserializable_handler_fn, path);
        }
        return json_obj;
    }

    if (py::isinstance<py::list>(source) || py::isinstance<py::tuple>(source))
    {
        const auto py_list = source.cast<py::list>();
        auto json_arr      = json_t::array();
        for (const auto& p : py_list)
        {
            std::string path{parent_path + "/" + std::to_string(json_arr.size())};
            json_arr.push_back(cast_from_pyobject_impl(p.cast<py::object>(), unserializable_handler_fn, path));
        }

        return json_arr;
    }

    if (py::isinstance<py::bool_>(source))
    {
        return json_t(py::cast<bool>(source));
    }

    if (py::isinstance<py::int_>(source))
    {
        return json_t(py::cast<long>(source));
    }

    if (py::isinstance<py::float_>(source))
    {
        return json_t(py::cast<double>(source));
    }

    if (py::isinstance<py::str>(source))
    {
        return json_t(py::cast<std::string>(source));
    }

    // source is not serializable, return as a binary object in PythonByteContainer
    return json_t::binary(PythonByteContainer(py::cast<mrc::pymrc::PyHolder>(source)), type_to_uint64<py::object>());

    // NOLINTEND(modernize-return-braced-init-list)
}

json_t cast_from_pyobject(const py::object& source, mrc::pymrc::unserializable_handler_fn_t unserializable_handler_fn)
{
    return cast_from_pyobject_impl(source, unserializable_handler_fn);
}

json_t cast_from_pyobject(const py::object& source)
{
    return cast_from_pyobject_impl(source, nullptr);
}

}  // namespace morpheus::utilities