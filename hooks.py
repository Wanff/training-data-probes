from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict

from utils import untuple

class StatefulHook(ABC):
    #Inspired by: https://gist.github.com/danesherbs/f39a311aa9c3d14c90650a0f66d1ab2a
    def __init__(self, name):
        self.name = name
        self.module = None
    
    @abstractmethod
    def __call__(self, module, input, output):
        pass

class ContextHook(ABC):
    #Inspired by: https://gist.github.com/danesherbs/91237e0b6e1534c7248377de549c875a

    def __init__(self, name, module):
        self.name = name
        self.module = module

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self.hook)
        return self

    def __exit__(self, type, value, traceback):
        self.handle.remove()

    @abstractmethod
    def hook(self, module, input, output):
        pass  

#Stateful Hooks
class InputHook(StatefulHook):
    """
    MCoefHook should be registered at transformer.h[x].mlp.c_proj 
    (gets the input of the second linear transformation in the MLP after the GELU/RELU)
    """
    def __init__(self, name):
        super().__init__(name)
        
        self.activations = None
        self.seq_len = None

    def __call__(self, module, input, output):
        self.module = module
        
        num_tokens = list(untuple(input).size())[1]  #shape: (batch, sequence, hidden_states)
        self.seq_len = num_tokens
        
        self.activations = untuple(input).detach()

class OutputHook(StatefulHook):
    """
      Attention should be registered at 
    """
    def __init__(self, name):
        super().__init__(name)
        
        self.activations = None
        self.seq_len = None

    def __call__(self, module, input, output):
        self.module = module
        
        # print(len(output))
        # print(output[0].shape)
        # print(len(output[1]))
        # print(untuple(output[1]).shape)
        # print(self.name)
        # print()
        num_tokens = list(untuple(output).size())[1]  #shape: (batch, sequence, hidden_states)
        self.seq_len = num_tokens
        
        self.activations = untuple(output).detach()
        
#Context Hooks

class AblateValuesHook(ContextHook):
    def __init__(self, name, module, values, coef_val = 0):
        super().__init__(name, module)
        self.values = values
        self.coef_val = coef_val
    
    def hook(self, module, input, output):
        output[:, :, self.values] = self.coef_val

    

        
        
        
        
        