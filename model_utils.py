import re
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import StatefulHook, InputHook, OutputHook
from utils import untuple

class ModelWrapper():
    """
    A wrapper for an autoregressive HF LM with hooking and activation storing functionality.
    Supports GPT2 and Pythia models.
    """
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_type = "pythia" if "pythia" in model_name else "gpt"
        
        if self.model_type == "pythia":
            self.num_layers = self.model.config.num_hidden_layers
        elif self.model_type == "gpt":
            self.num_layers = self.model.config.n_layer 
            
        self.hooks = {}
        self.save_ctx = {}

    def query_model_tok_dist(self, prompt: str, K: int = 10) -> List[Tuple[float, str]]:
        """
        Gets top 10 predictions and associated probabilities after last token in a prompt
        """
        tokens = self.tokenizer.encode_plus(prompt, return_tensors = 'pt').to(self.device)
        output = self.model(**tokens)
        logits = output['logits']
        
        trg_tok_idx = tokens['input_ids'].shape[1] - 1
        #gets probs after last tok in seq
        probs = F.softmax(untuple(logits)[0][trg_tok_idx], dim=-1) #the [0] is to index out of the batch idx
        probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

        #assert probs add to 1
        assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs)-1)) 

        probs_ = []
        for index, prob in enumerate(probs):
            probs_.append((index, prob))

        top_k = sorted(probs_, key = lambda x: x[1], reverse = True)[:K]
        top_k = [(t[1].item(), self.tokenizer.decode(t[0])) for t in top_k]
        
        return top_k

    def get_module(self, name):
        """
        Finds the named module within the given model.
        """
        for n, m in self.model.named_modules():
            if n == name:
                return m
        raise LookupError(name)
    
    def remove_all_hooks(self):
        for name, hook in self.hooks.items():
            hook.remove()
        
        self.hooks = {}
    
    def register_layer_hooks(self):
        for i in range(self.num_layers):
            if self.model_type == "pythia":
                layer_name = f"gpt_neox.layers.{i}"
            elif self.model_type == "gpt":
                layer_name = f"transformer.h.{i}"
            self.register_stateful_hook(layer_name, OutputHook(layer_name))
                
    def register_stateful_hook(self, module_name:str, stateful_hook:StatefulHook):
        module = self.get_module(module_name)
        
        self.hooks[stateful_hook.name] = module.register_forward_hook(stateful_hook) #saves the handle to the hooks dict
        self.save_ctx[stateful_hook.name] = stateful_hook #saves the activations to the save_ctx dict

    def register_dir_add_hook(self, module_name: str, intervention_idxs: List[int], dir : torch.Tensor, alpha: float = 1.0):
        def hook(module, input, output):
            for idx in intervention_idxs:
                output[0][:,idx,:] += dir * alpha
            return output
        
        module = self.get_module(module_name)
        self.hooks[module_name] =module.register_forward_hook(hook)