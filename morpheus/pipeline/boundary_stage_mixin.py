from abc import ABC


class BoundaryStageMixin(ABC):
    """
    Mixin intended to be added to both ingress and egress boundary stages, currently this only adds the ability to
    identify boundary stages.
    """
