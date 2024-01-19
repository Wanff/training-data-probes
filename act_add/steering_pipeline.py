import os
import re
import random
import json
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Union

import transformers
import torch
import torch.nn.functional as F

import openai
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

import plotly.graph_objects as go
import plotly.express as px

from utils import untuple

from act_add.model_wrapper import ModelWrapper
from act_add.rep_reader import RepReader, CAARepReader, PCARepReader
from act_add.contrast_dataset import ContrastDataset

class SteeringPipeline():
    def __init__(self, model_wrapper : ModelWrapper, contrast_dataset : ContrastDataset, rep_reader : RepReader):
        self.model_wrapper = model_wrapper
        self.contrast_dataset = contrast_dataset
        self.rep_reader = rep_reader
                            
        self.model_wrapper.wrap_all()
    def gen_dir_from_states(self, 
                            pos_hidden_states,
                            neg_hidden_states,
                            hidden_layers : Union[List[int], int] = -1,
                            n_difference : int = 1,
                            train_labels: List[int] = None,):
        
        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]
        
        assert pos_hidden_states.shape[1] == neg_hidden_states.shape[1], "pos and neg hidden states must have same number of examples"
        
        #*this is if shape is n_examples x n_layers x n_hidden
        # interweaved = [torch.stack([pos_hidden_states[i], neg_hidden_states[i]], dim = 0) for i in range(pos_hidden_states.shape[0])]
        # hidden_states = torch.cat(interweaved, dim=0)
        
        #*this is if shape is n_layers x n_examples x n_hidden
        interweaved = [torch.stack([pos_hidden_states[:, i], neg_hidden_states[:, i]], dim = 1) for i in range(pos_hidden_states.shape[1])]
        hidden_states = torch.cat(interweaved, dim=1)
        
        relative_hidden_states = self._gen_rel_states(hidden_states, hidden_layers, n_difference)
        
        return self._gen_dir(hidden_states, 
                             relative_hidden_states, 
                             hidden_layers, 
                             train_labels)
        
    def _gen_dir(self,       
                hidden_states,   
                relative_hidden_states,               
                hidden_layers : Union[List[int], int] = -1,
                train_labels: List[int] = None,
                        ):
        
        # get the directions
        directions = self.rep_reader.get_rep_directions(
            self.model_wrapper.model, self.model_wrapper.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels)

        for layer in self.rep_reader.directions:
            if type(self.rep_reader.directions[layer]) == np.ndarray:
                self.rep_reader.directions[layer] = self.rep_reader.directions[layer].astype(np.float32)

        self.rep_reader.direction_signs = self.rep_reader.get_signs(
            hidden_states, train_labels, hidden_layers)
        
        return self.rep_reader.directions
        
    def _gen_rel_states(self, hidden_states, hidden_layers, n_difference):
        #*hidden_states should be a tensor or tuple of tensors of shape (n_layers, n_examples, n_hidden)
        
        if isinstance(hidden_states, dict):
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            
        else:
            relative_hidden_states = {k: np.copy(hidden_states[k]) for k in range(hidden_states.shape[0])}
        
        if isinstance(self.rep_reader, PCARepReader):
            # get differences between pairs
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]
        elif isinstance(self.rep_reader, CAARepReader):
            #* IMPORTANT: All RepReaders expects that the order of the training data is alternating like: [p, n, p, n, ...]
                for layer in hidden_layers:
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]
        
        return relative_hidden_states
                        
    def gen_dir_from_strings(self, 
                        train_inputs: Union[str, List[str], List[List[str]]], 
                        rep_token_idx : int = -1, 
                        hidden_layers : Union[List[int], int] = -1,
                        n_difference : int = 1,
                        train_labels: List[int] = None,):
        self.model_wrapper.reset()
        
        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]

        # get raw hidden states for the train inputs
        hidden_states = self.model_wrapper.batch_hiddens(train_inputs, 
                                                        hidden_layers, 
                                                        rep_token_idx, 
                                                        )['resid']
        relative_hidden_states = self._gen_rel_states(hidden_states, hidden_layers, n_difference)
        
        return self._gen_dir(hidden_states, 
                             relative_hidden_states, 
                             hidden_layers, 
                             train_labels)

        
    def batch_steering_generate(self, 
                                inputs : List[str], 
                                layers_to_intervene : List[int],
                                coeff : float = 1.0,
                                token_pos : Union[str, int] = None,
                                batch_size=8, 
                                operator = "linear_comb",
                                use_tqdm=True,
                                **generation_kwargs,
                                ):
        
        assert self.rep_reader.directions is not None, "Must generate rep_reader directions first"
        
        #? do i need to do half() here?
        steering_vectors = {}
        for layer in layers_to_intervene:
            if isinstance(self.rep_reader.directions[layer], np.ndarray):
                steering_vectors[layer] = torch.tensor(coeff * self.rep_reader.directions[layer] * self.rep_reader.direction_signs[layer]).to(self.model_wrapper.model.device).half()
            else:
                steering_vectors[layer] = (coeff * self.rep_reader.directions[layer] * self.rep_reader.direction_signs[layer]).to(self.model_wrapper.model.device).half()

        self.model_wrapper.reset()
        self.model_wrapper.set_controller(layers_to_intervene, steering_vectors, masks=1, token_pos = token_pos, operator = operator)
        generated = []

        iterator = tqdm(range(0, len(inputs), batch_size)) if use_tqdm else range(0, len(inputs), batch_size)

        for i in iterator:
            inputs_b = inputs[i:i+batch_size]
            decoded_outputs = self.model_wrapper.batch_generate_autoreg(inputs_b, **generation_kwargs)
            decoded_outputs = [o.replace(i, "") for o,i in zip(decoded_outputs, inputs_b)]
            generated.extend(decoded_outputs)

        self.model_wrapper.reset()
        return generated
    
        