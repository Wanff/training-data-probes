#From: https://github.com/nrimsky/SycophancySteering/blob/main/utils/tokenize_llama.py

from typing import List, Tuple, Optional

import json
import random
from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import AutoTokenizer
from enum import Enum

# class PromptType(Enum):
#     LLAMA_CHAT_CONVO = 1
#     LLAMA_BASE_CONVO = 2
#     NO_CONVO = 3

class ContrastDataset():
    def __init__(self, p_prompts, n_prompts, tokenizer_name,
                 use_convo_format: bool = False,
                 format_fn = None, 
                 use_chat: bool = False, 
                 system_prompt: str = ""):
        
        self.p_prompts = p_prompts
        self.n_prompts = n_prompts
        
        if format_fn is not None:
            self.p_prompts = [format_fn(p) for p in self.p_prompts]
            self.n_prompts = [format_fn(n) for n in self.n_prompts]
                
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, 
                                                       use_fast=True, 
                                                       padding_side="left", 
                                                       legacy=False, 
                                                       token=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.system_prompt = system_prompt
        self.use_chat = use_chat
        
        if use_convo_format:
            assert len(self.system_prompt) > 0, "Must provide a system prompt if using convo format"
            assert (len(self.p_prompts[0]) == 2) and (len(self.n_prompts[0]) == 2), "Must provide a list of tuples of the form (user_input, model_output)"
            
            # self.p_prompts = self.tokenizer.batch_decode(tokenize_llama(self.tokenizer, 
            #                                                                  self.system_prompt, 
            #                                                                  self.p_prompts, 
            #                                                                  chat_model=self.use_chat)["input_ids"])
            
            # self.n_prompts = self.tokenizer.batch_decode(tokenize_llama(self.tokenizer, 
            #                                                                  self.system_prompt, 
            #                                                                  self.n_prompts, 
            #                                                                  chat_model=self.use_chat)["input_ids"])
          
            self.p_prompts = [self.tokens_to_prompt(tokenize_llama(self.tokenizer, 
                                             self.system_prompt, 
                                             [(user_in, model_out)], 
                                             chat_model=self.use_chat)) for (user_in, model_out) in self.p_prompts]
            
            self.n_prompts = [self.tokens_to_prompt(tokenize_llama(self.tokenizer, 
                                             self.system_prompt, 
                                             [(user_in, model_out)], 
                                             chat_model=self.use_chat)) for (user_in, model_out) in self.n_prompts]

        self.data = [[p,n] for p,n in zip(self.p_prompts, self.n_prompts)] #positive always comes first
        
    def prompt_to_tokens(self, prompt):
        tokens = self.tokenizer.encode(prompt)
        return torch.tensor(tokens).unsqueeze(0)
    
    def tokens_to_prompt(self, tokens):
        
        return self.tokenizer.decode(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p_text = self.p_prompts[idx]
        n_text = self.n_prompts[idx]

        p_tokens = self.prompt_to_tokens(p_text)
        n_tokens = self.prompt_to_tokens(n_text)
        return p_tokens, n_tokens
        
    def gen_train_test_split(self, train_size=0.8, seed = None):
        random.seed(seed)

        if train_size > 1:
            # number of examples
            if seed is None:
                train_indices = np.random.choice(len(self), train_size, replace=False)
            else:
                train_indices = list(range(train_size))
        else:
            #fraction of examples
            if seed is None:
                train_indices = np.random.choice(len(self), int(len(self) * train_size), replace=False)
            else:
                train_indices = list(range(int(len(self) * train_size)))
        
        labels = []
        for d in self.data:
            true_s = d[0]
            # random.shuffle(d)
            labels.append([s == true_s for s in d]) #looks like [[True, False], [False, True], ...]
        
        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in range(len(self)) if i not in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_labels = [labels[i] for i in range(len(self)) if i not in train_indices]

        #flatten everything to go from [[p, n]] to [p, n]
        train_data = np.concatenate(train_data).tolist() 
        test_data = np.concatenate(test_data).tolist()
        train_labels = np.concatenate(train_labels).tolist()
        test_labels = np.concatenate(test_labels).tolist()
        
        return train_data, test_data, train_labels, test_labels
    

#* Llama tokenization
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def tokenize_llama(
    tokenizer,
    system_prompt: str,
    conversation: List[Tuple[str, Optional[str]]],
    no_final_eos=False,
    chat_model=True,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    system_prompt: the system prompt to use for the conversation
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response
    chat_model: whether input is meant for a llama chat model or a llama base model (True means chat model, False means base model)

    Returns: a list of tokens
    """
    if chat_model:
        return tokenize_llama_chat(tokenizer, system_prompt, conversation, no_final_eos)
    else:
        return tokenize_llama_base(tokenizer, conversation, no_final_eos)


def tokenize_llama_chat(
    tokenizer,
    system_prompt: str,
    conversation: List[Tuple[str, Optional[str]]],
    no_final_eos=False,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    system_prompt: the system prompt to use for the conversation
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """

    def _instruction_response_to_tokens(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ):
        if is_first_message:
            dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
        else:
            dialog_content = instruction.strip()
        if model_output is not None:
            if no_eos:
                return tokenizer.encode(
                    f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                )
            return tokenizer.encode(
                f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()} {tokenizer.eos_token}"
            )
        else:
            return tokenizer.encode(f"{B_INST} {dialog_content.strip()} {E_INST}")

    tokens = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens

def tokenize_llama_base(
    tokenizer, conversation: List[Tuple[str, Optional[str]]], no_final_eos=False
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """
    if len(conversation) == 0:
        return []
    full_text = []
    for user_input, model_output in conversation:
        text = f"Input: {user_input.strip()}"
        if model_output:
            text += f"\nResponse: {model_output.strip()}"
        full_text.append(text)
    full_text = "\n\n".join(full_text)
    if not no_final_eos and conversation[-1][1] is not None:
        full_text += f" {tokenizer.eos_token}"
    return tokenizer.encode(full_text)

