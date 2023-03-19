import torch


class DistributedAutoEncoder(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, *args, **kwargs):
        self.__dict__['initialized'] = False

        # set placeholder attribute for the pytorch module, which will be 
        # populated in the init function of the super class
        self.module = None
        super().__init__(*args, **kwargs)
        
        self.__dict__['initialized'] = True
        
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __setattr__(self, name, value):
        # set the attribute to the instance it belongs to after init
        if not self.initialized or hasattr(self, name):
            super().__setattr__(name, value)
        else:
            setattr(self.module, name, value)
        