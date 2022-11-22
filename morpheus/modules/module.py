from abc import ABC, abstractmethod
import typing
from morpheus.config import Config
import srf
from srf.core import operators as ops


class Module(ABC):

    def __init__(self, config: Config, module_config: typing.Dict):

        self._config = config

        self._module_id = module_config["module_id"]
        self._version = module_config["version"]
        self._module_name = module_config["module_name"]
        self._module_namespace = module_config["module_namespace"]
        self._module_config = module_config

        self._registry = srf.ModuleRegistry()

    @abstractmethod
    def on_data():
        pass

    def register_module(self, unique_name: str):

        def module_init(builder: srf.Builder):

            def node_fn(obs: srf.Observable, sub: srf.Subscriber):
                obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

            node = builder.make_node_full(unique_name, node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(self._module_id, self._module_namespace):
            self._registry.register_module(self._module_id, self._module_namespace, self._version, module_init)

        fn_constructor = self._registry.get_module_constructor(self._module_id, self._module_namespace)
        module = fn_constructor(self._module_name, self._module_config)

        return module

