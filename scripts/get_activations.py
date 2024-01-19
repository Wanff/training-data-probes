import transformers
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import datasets
from datasets import load_dataset
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Union

from tqdm import tqdm
import argparse
import gc
import pickle

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

import sys
sys.path.append('../')  # Add the parent directory to the path
sys.path.append('../act_add')

from utils import untuple, char_by_char_similarity, tok_by_tok_similarity, levenshtein_distance
from act_add.model_wrapper import ModelWrapper, slice_acts

def gen_pile_data(N, tokenizer, min_n_toks : int = None):
    pile = datasets.load_dataset('EleutherAI/the_pile_deduplicated', split='train', streaming=True)
    sentences = []

    counter = 0
    for i, example in enumerate(pile):
        if min_n_toks is not None:
            toks = tokenizer(example['text'])['input_ids']
            if len(toks) > min_n_toks:
                sentences.append(example['text'])
                counter +=1
        else:
            sentences.append(example['text'])
            counter +=1
        
        if counter == N:
            break

    return sentences

def compare_token_lists(ground_toks, genned_toks):
    if len(ground_toks) != len(genned_toks):
        print(len(ground_toks), len(genned_toks))
        print("Both lists do not have the same length.")
        return 0
    
    num_same_tokens = sum(1 for token1, token2 in zip(ground_toks, genned_toks) if token1 == token2)
    percent_same_tokens = (num_same_tokens / len(ground_toks)) 
    
    return percent_same_tokens

def toks_to_string(tokenizer, toks):
    return "".join(tokenizer.batch_decode(toks))

# def slice_acts(out, N_TOKS: int, return_prompt_acts: bool, layers: List, tok_idxs: List = None):
#     """slices acts out of huggingface modeloutput object

#     Args:
#         out (_type_): _description_
#         N_TOKS (int): how many tokens generated
#         return_prompt_toks (bool): _description_
#         layers (List): _description_

#     Returns:
#         _type_: _description_
#     """
#     acts = torch.cat([torch.cat(out.hidden_states[i], dim = 0) for i in range(1, N_TOKS)], dim = 1)  #1, N_TOKS bc the first index is all previous tokens
#     #shape: n_layers + 1, N_TOKS - 1, d_M
#     #n_layers + 1 bc of embedding, N_TOKS - 1 bc of how max_new_tokens works
    
#     if return_prompt_acts:
#         prompt_acts = torch.cat(out.hidden_states[0], dim = 0)
#         acts = torch.cat([prompt_acts, acts], dim = 1)
    
#     acts = acts.cpu()
    
#     if tok_idxs is not None:
#         acts = acts[:, tok_idxs, :]
#     acts = acts[layers, :, :]
#     return acts

def ceildiv(a, b):
    #https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)

def get_memmed_activations_from_pregenned(mw: ModelWrapper, prompts: Union[List[str],List[int]],
                                          save_path: str,
                                          act_types: List[str] = ['resid'],
                                          save_every: int = 100,
                                          layers: List[int] = None,
                                           tok_idxs: List[int] = None,
                                           logging: bool = False,
                                           file_spec: str = "",
                                           **generation_kwargs,
                                          ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if layers is None:
        layers = list(range(mw.model.config.num_hidden_layers))
        
    acts_dict = {}
    for act_type in act_types:
        acts_dict[act_type] = dict([(layer, []) for layer in layers])
            
    for batch_idx in range(ceildiv(len(prompts), save_every) ):
        if logging:
            start_time = time.time()
            print(f"Batch {batch_idx + 1} of {len(prompts) // save_every}")
            
        batch = prompts[batch_idx * save_every : (batch_idx + 1) * save_every]
                
        batch_act_dict = mw.batch_hiddens(
                    batch,
                    layers = layers,
                    tok_idxs = tok_idxs,
                    return_types = act_types,
                    **generation_kwargs,
                    )

        for act_type in act_types:
            for layer in layers:
                acts_dict[act_type][layer].append(batch_act_dict[act_type][layer])
            
        if logging:
            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time
            print(f"Time elapsed for this batch: {elapsed_time:.2f} seconds")
            print(f"Num Hidden States Genned: {len(batch)}")
            for act_type in act_types:
                print(f"Shape of {act_type} one batch Hidden States: {acts_dict[act_type][layers[0]][-1].shape}")
            print()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Save all_hidden_states
        torch.save(acts_dict, save_path + f"/{file_spec}acts_dict.pt")
    
    for act_type in act_types:
        for layer in layers:
            acts_dict[act_type][layer] = torch.cat(acts_dict[act_type][layer], dim = 0)
            
    torch.save(acts_dict, save_path + f"/{file_spec}acts_dict.pt")
    
    gc.collect()
    torch.cuda.empty_cache()
    return acts_dict

def get_memmed_activations(mw: ModelWrapper, prompts: Union[List[str],List[int]],
                    save_path: str,
                    save_every: int = 100, 
                    N_TOKS: int = 32,
                    layers: List[int] = None,
                    tok_idxs: List[int] = None,
                    return_prompt_acts: bool = False,
                    logging: bool = False,
                    file_spec: str = "",
                    **generation_kwargs):
    """
    autoregressive generation for resid activations

    Args:
        mw (ModelWrapper): _description_
        prompts (Union[List[str],List[int]]): _description_
        save_path (str): _description_
        act_types (List[str], optional): _description_. Defaults to ['resid'].
        save_every (int, optional): _description_. Defaults to 100.
        check_if_memmed (bool, optional): _description_. Defaults to True.
        N_TOKS (int, optional): _description_. Defaults to 32.
        layers (List[int], optional): _description_. Defaults to None.
        tok_idxs (List[int], optional): _description_. Defaults to None.
        return_prompt_acts (bool, optional): _description_. Defaults to False.
        logging (bool, optional): _description_. Defaults to False.
        file_spec (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    if not os.path.exists(save_path):
        os.makedirs(args.save_path)
    
    if layers is None:
        layers = range(1, mw.model.config.num_hidden_layers + 1)
        
    all_generations = []
    all_tokens = []
    all_hidden_states = []
    all_mem_status = {
        "lev_distance": [],
        "char_by_char_sim": [],
        "tok_by_tok_sim": [],
    }
    
    for batch_idx in range(ceildiv(len(prompts), save_every) ):
        if logging:
            start_time = time.time()
            print(f"Batch {batch_idx + 1} of {ceildiv(len(prompts), save_every)}")
            
        batch = prompts[batch_idx * save_every : (batch_idx + 1) * save_every]
        
        if isinstance(prompts[0], str):
            raise NotImplementedError("Need to implement batch_generate_autoreg for str prompts")
        
        out = mw.batch_generate_autoreg(prompts=batch[:, :N_TOKS],
                    max_new_tokens=N_TOKS,
                    output_hidden_states=True,
                    output_tokens=True,
                    layers = layers,
                    tok_idxs = tok_idxs,
                    return_prompt_acts=return_prompt_acts,
                    top_p = 1.0,
                    **generation_kwargs,
                    )
        
        all_hidden_states.extend(out['hidden_states'].cpu())
        all_generations.extend(out['generations'])
        all_tokens.extend(out['tokens'].cpu().numpy().tolist())
        
        print(out['hidden_states'].shape)
    
        all_mem_status['tok_by_tok_sim'].extend(tok_by_tok_similarity(all_tokens[-len(batch):], batch))
        all_mem_status['char_by_char_sim'].extend(char_by_char_similarity(mw.tokenizer.batch_decode(batch), out['generations']))
        all_mem_status['lev_distance'].extend(levenshtein_distance(mw.tokenizer.batch_decode(batch), out['generations']))
    
        if logging:
            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time
            print(f"Time elapsed for this batch: {elapsed_time:.2f} seconds")
            print(f"Num Hidden States Genned: {len(batch)}")
            print(f"Shape of Hidden States: {all_hidden_states[-1].shape}")
            print(f"Generations:")
            for i, g in enumerate(all_generations[-len(batch):]):
                print(g)
                print(f"Lev {all_mem_status['lev_distance'][batch_idx*save_every + i]}")
                print(f"Char by Char {all_mem_status['char_by_char_sim'][batch_idx*save_every + i]}")
                print(f"Tok by Tok {all_mem_status['tok_by_tok_sim'][batch_idx*save_every + i]}")
                print()
            print()
        
        gc.collect()
        torch.cuda.empty_cache()

        #save hidden_states
        torch.save(all_hidden_states[-len(batch):], save_path + f"/{file_spec}check{batch_idx}_all_hidden_states.pt")

        # Save all_generations
        with open(save_path + f"/{file_spec}all_generations.pkl", "wb") as f:
            pickle.dump(all_generations, f)
            
        with open(save_path + f"/{file_spec}all_tokens.pkl", "wb") as f:
            pickle.dump(all_tokens, f)
            
        # Save all_mem_status
        with open(save_path + f"/{file_spec}all_mem_status.pkl", "wb") as f:
            pickle.dump(all_mem_status, f)
    
    all_hidden_states = torch.stack(all_hidden_states, dim = 0)
    torch.save(all_hidden_states, save_path + f"/{file_spec}all_hidden_states.pt")
    
    return all_hidden_states, all_generations, all_tokens, all_mem_status
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--act_types",
        nargs="+",
        default = None,
    )
    parser.add_argument(
        "--N_PROMPTS",
        type=int,
        required=True,
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--N_TOKS",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--return_prompt_acts",
        action="store_true",
    )
    parser.add_argument(
        "--logging",
        action="store_true",
    )

    args = parser.parse_args()
    
    print(args)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    mw = ModelWrapper(model, tokenizer)
    
    #* autoreg llama generation run
    llama_ground_data = pd.read_csv(f'../data/llama-2-7b/llama_ground_data.csv')
    print(llama_ground_data.head())
    
    tok_idxs = (7 * np.arange(10)).tolist() #every 5th token
    tok_idxs[-1]= - 1 #goes from 63 to 62
    tok_idxs[0] = 1
    print(tok_idxs)
    ground_tokens = [eval(toks) for toks in llama_ground_data['ground_tokens'].values.tolist()][:args.N_PROMPTS]
    print(ground_tokens)
    print(np.array(ground_tokens).shape)
    all_hidden_states, all_generations, all_tokens, all_mem_status = get_memmed_activations(mw, 
                                                                            torch.tensor(ground_tokens), 
                                                                            args.save_path,
                                                                            save_every = args.save_every,
                                                                            N_TOKS = args.N_TOKS,
                                                                            layers = args.layers,
                                                                            tok_idxs = tok_idxs,
                                                                            return_prompt_acts = args.return_prompt_acts,
                                                                            logging = args.logging,
                                                                            file_spec = "")
    
    
    
    #* mlp/attn from pregenned
    # all_mem_12b_data = pd.read_csv(f'data/12b/mem_evals_gen_data.csv')
    
    # prompts = all_mem_12b_data[all_mem_12b_data['source'] == 'pythia-evals'][:args.N_PROMPTS].gen.values.tolist() + all_mem_12b_data[all_mem_12b_data['source'] == 'pile'][:args.N_PROMPTS].gen.values.tolist()
    
    # tok_idxs = (7 * np.arange(10)).tolist() #every 5th token
    # tok_idxs[-1]= - 1 #goes from 63 to 62
    # acts_dict = get_memmed_activations_from_pregenned(mw,
    #                                                   prompts,
    #                                                   args.save_path,
    #                                                   act_types = args.act_types,
    #                                                   save_every = args.save_every,
    #                                                   layers = args.layers,
    #                                                   tok_idxs = tok_idxs,
    #                                                   logging = args.logging,
    #                                                   file_spec = "attn_mlp_from_pregenned")
    
    #* autoreg pythia generation run
    # if "deduped" in args.model_name:
    #     #model_name looks like: EleutherAI/pythia-12B-deduped
    #     dataset_name = "deduped." + args.model_name.split("-")[-2]
    # else:
    #     #model_name looks like: EleutherAI/pythia-12B
    #     dataset_name = "duped." + args.model_name.split("-")[-1]
    
    # mem_data = load_dataset('EleutherAI/pythia-memorized-evals')[dataset_name]

    # mem_data_toks = [seq for seq in mem_data[:args.N_PROMPTS]['tokens']]
    
    # # mem_data_prompts = [toks_to_string(tokenizer, seq) for seq in mem_data_toks]
    
    # pile_prompts = gen_pile_data(args.N_PROMPTS, tokenizer, min_n_toks = 64)
    # print(len(pile_prompts))
    # print(len(mem_data_toks))
    
    # tok_idxs =  (7 * np.arange(10)).tolist() #every 5th token
    # tok_idxs[-1]= tok_idxs[-1] - 1 #goes from 63 to 62
    # print(tok_idxs)
    # mem_hidden_states, mem_generations, mem_mem_status = get_memmed_activations(model, 
    #                                                                             tokenizer, 
    #                                                                             mem_data_toks, 
    #                                                                             args.save_path,
    #                                                                             save_every = args.save_every,
    #                                                                             check_if_memmed = args.check_if_memmed,
    #                                                                             N_TOKS = args.N_TOKS,
    #                                                                             layers = args.layers,
    #                                                                             tok_idxs = tok_idxs,
    #                                                                             return_prompt_acts = args.return_prompt_acts,
    #                                                                             logging = args.logging,
    #                                                                             file_spec = "mem_")
    
    # pile_hidden_states, pile_generations, pile_mem_status = get_memmed_activations(model,
    #                                                                                 tokenizer, 
    #                                                                                 pile_prompts, 
    #                                                                                 args.save_path,
    #                                                                                 save_every = args.save_every,
    #                                                                                 check_if_memmed = args.check_if_memmed,
    #                                                                                 N_TOKS = args.N_TOKS,
    #                                                                                 layers = args.layers,
    #                                                                                 tok_idxs = tok_idxs,
    #                                                                                 return_prompt_acts = args.return_prompt_acts,
    #                                                                                 logging = args.logging,
    #                                                                                 file_spec = "pile_")
    
    
    