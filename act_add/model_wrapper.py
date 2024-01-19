#from: https://github.com/andyzoujm/representation-engineering/blob/main/repe/rep_control_reading_vec.py
# wrapping classes
import torch
import numpy as np
from typing import List, Union, Optional
from einops import rearrange

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
            
            self.controller = self.controller.to(modified.device)
            if type(mask) == torch.Tensor:
                mask = mask.to(modified.device)
            if isinstance(self.token_pos, int):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple) or isinstance(self.token_pos, np.ndarray):
                if self.controller.shape[0] > 1 and modified.shape[1] == 1:
                    #if controller is multiple tokens and modified is one, meaning we are in autoregressive generation mode, we skip out of this loop
                    pass
                else:
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

                assert len(self.controller.shape) == len(modified.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."
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
                # print(current.shape, controller.shape)
                projection = torch.sum(current.float() * controller.float(), dim = 2).unsqueeze(2) * controller.float()
                if current.dtype == torch.float16:
                    projection = projection.half()
                return current - projection
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


#! attention always comes first, mlp comes second bc that's just how it is
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

GPT_BLOCK_NAMES = [
    'attn',
    'mlp',
    'ln_1',
    'ln_2'
]

def slice_acts(out, N_TOKS: int, return_prompt_acts: bool, layers: List, tok_idxs: List = None):
    """slices acts out of huggingface modeloutput object

    Args:
        out (_type_): _description_
        N_TOKS (int): how many tokens generated
        return_prompt_toks (bool): _description_
        layers (List): _description_

    Returns:
        _type_: _description_
    """
    #! expects hf layer idxs (ie 1-index)
    #out.hidden_states is shape  max_new_tokens x n_layers + 1 x batch x activations
    
    #first loop goes through the tokens, second loop goes through the layers or something
    acts = torch.stack([torch.cat(out.hidden_states[i], dim = 1) for i in range(1, N_TOKS)], dim = 1)  #1, N_TOKS bc the first index is all previous tokens
    #shape: batch_size x N_TOKS - 1 x n_layers + 1 x d_M
    #n_layers + 1 bc of embedding, N_TOKS - 1 bc of how max_new_tokens works
    acts = rearrange(acts, 'b t l d -> b l t d')
    
    if return_prompt_acts:
        prompt_acts = torch.stack(out.hidden_states[0], dim = 0) #shape: n_layers + 1 x batch_size x seq_len x d_M
        prompt_acts = rearrange(prompt_acts, 'l b t d -> b l t d')
        acts = torch.cat([prompt_acts, acts], dim = 2)
    
    acts = acts.cpu()
    # print(acts.shape)
    
    if tok_idxs is not None:
        acts = acts[:, :, tok_idxs, :]
    acts = acts[:, layers, :, :]
    return acts

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
        elif hasattr(self.model, 'transformer'):
            self.block_names = GPT_BLOCK_NAMES
            self.model_base = self.model.transformer
            self.model_base.layers = self.model.transformer.h

        
    #Generation Functions
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
        
    def batch_generate_autoreg(self, prompts, 
                                   max_new_tokens: int = 32,
                                   output_hidden_states = False, 
                                   output_tokens = False, 
                                   layers = None,
                                   tok_idxs = None,
                                    return_prompt_acts = False,
                                   **kwargs):
        #slow iterative version bleh:
        # generations = []
        # for s in strings:
        #     assert isinstance(s, str), "Input must be a list of strings."
            
        #     toks = self.tokenizer(s, return_tensors="pt", padding=True, max_length=512, truncation=True)
        #     input_ids = toks.input_ids.to(self.model.device)
        #     attention_mask = toks.attention_mask.to(self.model.device)
            
        #     out = self.model.generate(
        #                 input_ids=input_ids,
        #                 attention_mask=attention_mask,
        #                 pad_token_id=self.tokenizer.eos_token_id,
        #                 **kwargs
        #                 )
        #     generation = self.tokenizer.decode(out[0], skip_special_tokens=True)
        #     generations.append(generation)
        
        if isinstance(prompts[0], str):
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, max_length=512, truncation=True)
        else:
            inputs = self.tokenizer.pad({'input_ids': prompts}, padding = True, return_attention_mask=True)
            
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens = max_new_tokens,
                    output_hidden_states = output_hidden_states,
                    return_dict_in_generate = output_hidden_states,
                    **kwargs
                    )
        
        #! this is not good code:
        if output_hidden_states and output_tokens:
            assert layers is not None and tok_idxs is not None, "Must specify layers and token indices to slice."
            return {"generations": self.tokenizer.batch_decode(out['sequences'], skip_special_tokens=True),
                    "tokens": out['sequences'],
                    "hidden_states": slice_acts(out, 
                                                N_TOKS = max_new_tokens, 
                                                layers = layers,
                                                tok_idxs = tok_idxs,
                                                return_prompt_acts = return_prompt_acts),
                    }      
        else:
            return self.tokenizer.batch_decode(out, skip_special_tokens=True)            
        
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
            tok_idxs: Union[List[int], int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']
    
        hidden_states_layers = {}
        for layer in hidden_layers:
            layer = layer + 1 #we do this because usually we 0-index layers, but hf does not
            hidden_states = outputs['hidden_states'][layer]
            hidden_states =  hidden_states[:, tok_idxs, :]
            # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
            hidden_states_layers[layer - 1] = hidden_states.detach().cpu().to(dtype = torch.float32)

        return hidden_states_layers
    
    def batch_hiddens(self, 
                      prompts: Union[List[str], List[int]], 
                      layers: List[int], 
                      tok_idxs: Union[List[int], int] = -1, 
                    return_types = ['resid'], **kwargs):
        """
        Takes a list of strings or tokens and returns the hidden states of the specified layers and token indices.
        
        does not support autoregressive generation
        """
        self.reset()
        self.wrap_all()
        hidden_states_dict = {}
        
        with torch.no_grad():
            # self.reset()
            
            if isinstance(prompts[0], str):
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, max_length=512, truncation=True)

            else:
                inputs = self.tokenizer.pad({'input_ids': prompts}, padding = True, return_attention_mask=True)
                
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            
            outputs = self.model(input_ids = input_ids, attention_mask=attention_mask, output_hidden_states = True, **kwargs)

            for act_type in return_types:
                if act_type == 'resid':
                    hidden_states_dict['resid'] = self._get_hidden_states(outputs, tok_idxs, layers, 'residual') #this is layers to acts
                elif act_type == 'attn':
                    hidden_states_dict['attn'] = self.get_activations(layers, self.block_names[0])
                    for layer in layers:
                        hidden_states_dict['attn'][layer] = hidden_states_dict['attn'][layer][:, tok_idxs, :]
                elif act_type == 'mlp':
                    hidden_states_dict['mlp'] = self.get_activations(layers, self.block_names[1])
                    for layer in layers:
                        hidden_states_dict['mlp'][layer] = hidden_states_dict['mlp'][layer][:, tok_idxs, :]
                else:
                    assert False, f"Unknown activation type {act_type}."
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
                    return current_layer.output.detach().cpu()
                elif block_name in self.block_names and self.is_wrapped(getattr(current_block, block_name)):
                    return getattr(current_block, block_name).output.detach().cpu()
                else:
                    assert False, f"No wrapped block named {block_name}."

            else:
                if block_name in self.block_names and self.is_wrapped(getattr(current_layer, block_name)):
                    return getattr(current_layer, block_name).output.detach().cpu()
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
      