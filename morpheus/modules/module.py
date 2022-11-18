from abc import ABC, abstractmethod
import typing
import srf
from srf.core import operators as ops
import functools


class Module(ABC):

    def __init__(self,
                 config: typing.Dict,
                 version: typing.List,
                 module_id: str,
                 module_name: str,
                 module_namespace: str):

        self._module_id = module_id
        self._version = version
        self._module_name = module_name
        self._module_namespace = module_namespace

        self._registry = srf.ModuleRegistry
        self._builder = srf.Builder

    @abstractmethod
    def on_data():
        pass

    def register_module(func):

        @functools.wraps(func)
        def register(*args, **kwargs):

            def module_init(builder: srf.Builder):

                def node_fn(obs: srf.Observable, sub: srf.Subscriber):
                    obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

                node = builder.make_node_full(self.unique_name, node_fn)

                builder.register_module_input("input", node)
                builder.register_module_output("output", node)

            if not self._registry.contains(self._module_id, self._module_namespace):
                self._registry.register_module(self._module_id, self._module_namespace, self._version, module_init)
            obj, type = func(**args, **kwargs)

        return register
