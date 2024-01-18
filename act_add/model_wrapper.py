#from: https://github.com/andyzoujm/representation-engineering/blob/main/repe/rep_control_reading_vec.py
# wrapping classes
import torch
import numpy as np
from typing import List, Union, Optional

class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.normalize = False

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output

            
        if self.controller is not None:
        
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)

            if self.mask is not None:
                mask = self.mask

            # we should ignore the padding tokens when doing the activation addition
            # mask has ones for non padding tokens and zeros at padding tokens.
            # only tested this on left padding
            elif "position_ids" in kwargs:
                pos = kwargs["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                target_shape = modified.shape
                mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)
                mask = mask.to(modified.dtype)
            else:
                # print(f"Warning: block {self.block_name} does not contain information 'position_ids' about token types. When using batches this can lead to unexpected results.")
                mask = 1.0

            if len(self.controller.shape) == 1:
                self.controller = self.controller.reshape(1, 1, -1)
            assert len(self.controller.shape) == len(modified.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."

            self.controller = self.controller.to(modified.device)
            if type(mask) == torch.Tensor:
                mask = mask.to(modified.device)
            if isinstance(self.token_pos, int):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple) or isinstance(self.token_pos, np.ndarray):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, str):
                if self.token_pos == "end":
                    len_token = self.controller.shape[1]
                    modified[:, -len_token:] = self.operator(modified[:, -len_token:], self.controller * mask)
                elif self.token_pos == "start":
                    len_token = self.controller.shape[1]
                    modified[:, :len_token] = self.operator(modified[:, :len_token], self.controller * mask)
                else:
                    assert False, f"Unknown token position {self.token_pos}."
            else:
                modified = self.operator(modified, self.controller * mask)

            if self.normalize:
                norm_post = torch.norm(modified, dim=-1, keepdim=True)
                modified = modified / norm_post * norm_pre
            
        if isinstance(output, tuple):
            output = (modified,) + output[1:] 
        else:
            output = modified
        
        return output

    def set_controller(self, activations, token_pos=None, masks=None, normalize=False, operator='linear_comb'):
        self.normalize = normalize
        self.controller = activations.squeeze()
        self.mask = masks
        self.token_pos = token_pos
        if operator == 'linear_comb':
            def op(current, controller):
                return current + controller
        elif operator == 'piecewise_linear':
            def op(current, controller):
                sign = torch.sign((current * controller).sum(-1, keepdim=True))
                return current + controller * sign
        elif operator == 'projection':
            def op(current, controller):
                raise NotImplementedError
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")
        self.operator = op
        
    def reset(self):
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.operator = None

    def set_masks(self, masks):
        self.mask = masks


LLAMA_BLOCK_NAMES = [
    "self_attn",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm"
    ]

PYTHIA_BLOCK_NAMES = [
    "attention",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm" 
]
    
def rename_attribute(object_, old_attribute_name, new_attribute_name):
    setattr(object_, new_attribute_name, getattr(object_, old_attribute_name))
    delattr(object_, old_attribute_name)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        self.model_base = self.model
        if hasattr(self.model, 'model'):
            self.block_names = LLAMA_BLOCK_NAMES
            self.model_base = self.model.model
        elif hasattr(self.model, 'gpt_neox'):
            self.block_names = PYTHIA_BLOCK_NAMES
            self.model_base = self.model.gpt_neox

        
    #Generation Functions
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
        
    def batch_generate_from_string(self, strings, **kwargs):
        generations = []
        for s in strings:
            assert isinstance(s, str), "Input must be a list of strings."
            
            toks = self.tokenizer(s, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = toks.input_ids.to(self.model.device)
            attention_mask = toks.attention_mask.to(self.model.device)
            
            out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                        )
            generation = self.tokenizer.decode(out[0], skip_special_tokens=True)
            generations.append(generation)
        
        return generations            
        
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.model.device)).logits
            return logits
        
    def run_prompt(self, prompt, **kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            return output
        
    def _get_hidden_states(
            self, 
            outputs,
            rep_token: Union[str, int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']
    
        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            hidden_states =  hidden_states[:, rep_token, :]
            # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
            hidden_states_layers[layer] = hidden_states.detach().cpu().to(dtype = torch.float32)

        return hidden_states_layers
    
    def batched_string_to_hiddens(self, strings: List[str], layers: List[int], token_idx: int = -1, **kwargs):
        """
        Takes a list of strings and returns the hidden states of the specified layers and token indices.
        in cpu torch
        """
        with torch.no_grad():
            # self.reset()
            
            inputs = self.tokenizer(strings, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            outputs = self.model(input_ids = input_ids, attention_mask=attention_mask, output_hidden_states = True)

            hidden_states_dict = self._get_hidden_states(outputs, token_idx, layers)        

            #* this also works, wrote it for my own understanding
            # temp_activations = self.get_activations(layers)
            # hidden_states_dict = {}
            # for layer in layers:
            #     hidden_states = temp_activations[layer][:, token_idx, :]
            #     if layer == -1:
            #         hidden_states = self.model_base.norm(hidden_states) #wowowow
            #     hidden_states_dict[layer] = hidden_states.detach().cpu().to(dtype = torch.float32)
                
            #! so the padding is appended to beginning not end, like in gpt, so doing token_idx = -1 is actually fine
            # last_token_indices = [len(self.tokenizer.tokenize(s)) - 1 for s in strings]
            # for layer in layers:
            #     activations[layer] = activations[layer][range(len(strings)), last_token_indices, :].cpu()
            # for layer in layers:
            #     activations[layer] = activations[layer][:, token_idx, :].cpu()      
            return hidden_states_dict
            
    #Wrapping Logic
    def wrap(self, layer_id, block_name):
        assert block_name in self.block_names
        if self.is_wrapped(self.model_base.layers[layer_id]):
            block = getattr(self.model_base.layers[layer_id].block, block_name)
            if not self.is_wrapped(block):
                setattr(self.model_base.layers[layer_id].block, block_name, WrappedBlock(block))
        else:
            block = getattr(self.model_base.layers[layer_id], block_name)
            if not self.is_wrapped(block):
                setattr(self.model_base.layers[layer_id], block_name, WrappedBlock(block))

    def wrap_decoder_block(self, layer_id):
        block = self.model_base.layers[layer_id]
        if not self.is_wrapped(block):
            self.model_base.layers[layer_id] = WrappedBlock(block)

    def wrap_all(self):
        for layer_id, layer in enumerate(self.model_base.layers):
            for block_name in self.block_names:
                self.wrap(layer_id, block_name)
            self.wrap_decoder_block(layer_id)
            
    def wrap_block(self, layer_ids, block_name):
        def _wrap_block(layer_id, block_name):
            if block_name in self.block_names:
                self.wrap(layer_id, block_name)
            elif block_name == 'decoder_block':
                self.wrap_decoder_block(layer_id)
            else:
                assert False, f"No block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _wrap_block(layer_id, block_name)
        else:
            _wrap_block(layer_ids, block_name)
        
    def reset(self):
        for layer in self.model_base.layers:
            if self.is_wrapped(layer):
                layer.reset()
                for block_name in self.block_names:
                    if self.is_wrapped(getattr(layer.block, block_name)):
                        getattr(layer.block, block_name).reset()
            else:
                for block_name in self.block_names:
                    if self.is_wrapped(getattr(layer, block_name)):
                        getattr(layer, block_name).reset()

    def set_masks(self, masks):
        for layer in self.model_base.layers:
            if self.is_wrapped(layer):
                layer.set_masks(masks)
                for block_name in self.block_names:
                    if self.is_wrapped(getattr(layer.block, block_name)):
                        getattr(layer.block, block_name).set_masks(masks)
            else:
                for block_name in self.block_names:
                    if self.is_wrapped(getattr(layer, block_name)):
                        getattr(layer, block_name).set_masks(masks)

    def is_wrapped(self, block):
        if hasattr(block, 'block'):
            return True
        return False
    
    def unwrap(self):
        for l, layer in enumerate(self.model_base.layers):
            if self.is_wrapped(layer):
                self.model_base.layers[l] = layer.block
            for block_name in self.block_names:
                if self.is_wrapped(getattr(self.model_base.layers[l], block_name)):
                    setattr(self.model_base.layers[l],
                            block_name,
                            getattr(self.model_base.layers[l], block_name).block)

    #Activation Storing and Interventions
    def get_activations(self, layer_ids, block_name='decoder_block'):

        def _get_activations(layer_id, block_name):
            current_layer = self.model_base.layers[layer_id]

            if self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name == 'decoder_block':
                    return current_layer.output
                elif block_name in self.block_names and self.is_wrapped(getattr(current_block, block_name)):
                    return getattr(current_block, block_name).output
                else:
                    assert False, f"No wrapped block named {block_name}."

            else:
                if block_name in self.block_names and self.is_wrapped(getattr(current_layer, block_name)):
                    return getattr(current_layer, block_name).output
                else:
                    assert False, f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            activations = {}
            for layer_id in layer_ids:
                activations[layer_id] = _get_activations(layer_id, block_name)
            return activations
        else:
            return _get_activations(layer_ids, block_name)


    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=None, masks=None, normalize=False, operator='linear_comb'):

        def _set_controller(layer_id, activations, block_name, masks, normalize, operator):
            current_layer = self.model_base.layers[layer_id]

            if block_name == 'decoder_block':
                current_layer.set_controller(activations, token_pos, masks, normalize, operator)
            elif self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name in self.block_names and self.is_wrapped(getattr(current_block, block_name)):
                    getattr(current_block, block_name).set_controller(activations, token_pos, masks, normalize, operator)
                else:
                    return f"No wrapped block named {block_name}."

            else:
                if block_name in self.block_names and self.is_wrapped(getattr(current_layer, block_name)):
                    getattr(current_layer, block_name).set_controller(activations, token_pos, masks, normalize, operator)
                else:
                    return f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            assert isinstance(activations, dict), "activations should be a dictionary"
            for layer_id in layer_ids:
                _set_controller(layer_id, activations[layer_id], block_name, masks, normalize, operator)
        else:
            _set_controller(layer_ids, activations, block_name, masks, normalize, operator)
      