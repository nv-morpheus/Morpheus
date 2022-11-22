from abc import ABC, abstractmethod
import typing
from morpheus.config import Config
import srf
from srf.core import operators as ops


class AbstractModule(ABC):

    def __init__(self, c: Config, mc: typing.Dict):

        self._c = c
        self._module_id = mc["module_id"]
        self._version = mc["version"]
        self._module_name = mc["module_name"]
        self._module_ns = mc["module_namespace"]
        self._mc = mc

        self._registry = srf.ModuleRegistry()

    @abstractmethod
    def on_data():
        pass

    # Registers a module
    def register_module(self):

        def module_init(builder: srf.Builder):

            def node_fn(obs: srf.Observable, sub: srf.Subscriber):
                obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

            node = builder.make_node_full(self._mc["unique_name"], node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(self._module_id, self._module_ns):
            self._registry.register_module(self._module_id, self._module_ns, self._version, module_init)

    # Registers modules which contains chain of inner modules
    def _register_chained_module(self):

        def module_init(builder: srf.Builder):

            prev_module = None
            head_module = None

            modules_conf = self._mc["modules"]

            for key in modules_conf.keys():
                module_conf = modules_conf[key]
                module = builder.load_module(module_conf["module_id"],
                                             module_conf["module_namespace"],
                                             module_conf["module_name"],
                                             module_conf)
                if prev_module:
                    builder.make_edge(prev_module.output_port("output"), module.input_port("input"))
                else:
                    head_module = module

                prev_module = module

            # Register module input and ouptut ports.
            builder.register_module_input("input", head_module.input_port("input"))
            builder.register_module_output("output", prev_module.output_port("output"))

        self._registry.register_module(self._module_id, self._module_ns, self._version, module_init)
