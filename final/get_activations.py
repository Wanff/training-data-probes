import transformers
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import datasets
from datasets import load_dataset, concatenate_datasets
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Union

from tqdm import tqdm
import argparse
import gc
import pickle

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_from_disk

import sys
sys.path.append('../')  # Add the parent directory to the path
sys.path.append('../act_add')

from utils import untuple, char_by_char_similarity, tok_by_tok_similarity, levenshtein_distance
from act_add.model_wrapper import ModelWrapper, slice_acts

def gen_pile_data(N, tokenizer, min_n_toks : int = None, split='train'): 

    if split == 'train': 
        pile = datasets.load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
    elif split == 'validation':
        pile = datasets.load_dataset('monology/pile-uncopyrighted', split='validation', streaming=True)
    else: 
        raise Exception("Invalid split")

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
        if batch_idx == int(.25 * ceildiv(len(prompts), save_every)) or batch_idx == int(.5 * ceildiv(len(prompts), save_every)) or batch_idx == int(.75 * ceildiv(len(prompts), save_every)):
            print("Saving batch")
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
                    save_mem_only: bool = False,
                    save_unmem_only: bool = False,
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
    
    assert not (save_mem_only and save_unmem_only), "Must save either mem or unmem"
    
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
    
    for batch_idx in tqdm(range(ceildiv(len(prompts), save_every) )):
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
                    do_sample = False,
                    **generation_kwargs,
                    )
        
        all_generations.extend(out['generations'])
        all_tokens.extend(out['tokens'].cpu().numpy().tolist())
        
        all_mem_status['tok_by_tok_sim'].extend(tok_by_tok_similarity(all_tokens[-len(batch):], batch))
        all_mem_status['char_by_char_sim'].extend(char_by_char_similarity(mw.tokenizer.batch_decode(batch ), out['generations']))
        all_mem_status['lev_distance'].extend(levenshtein_distance(mw.tokenizer.batch_decode(batch), out['generations']))
        
        if not save_mem_only and not save_unmem_only:
            all_hidden_states.extend(out['hidden_states'].cpu())
        else:
            if save_mem_only:
                mem_batch_idxs = [i for i, mem in enumerate(all_mem_status['tok_by_tok_sim'][-len(batch):]) if mem == 1]
            elif save_unmem_only:
                mem_batch_idxs = [i for i, mem in enumerate(all_mem_status['tok_by_tok_sim'][-len(batch):]) if mem < 0.6]
            memmed_states = out['hidden_states'].cpu()[mem_batch_idxs]
            
            all_hidden_states.extend(memmed_states)
            
            if logging:
                print(f"Num {'un' if save_unmem_only else ''}Memmed found {len(mem_batch_idxs)}")
                
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
        if not save_mem_only and not save_unmem_only:
            torch.save(all_hidden_states[-len(batch):], save_path + f"/{file_spec}check{batch_idx}_all_hidden_states.pt")
        else:
            torch.save(all_hidden_states[-len(mem_batch_idxs):], save_path + f"/{file_spec}check{batch_idx}_all_hidden_states.pt")

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
    
    #* delete the checkpoints
    print("deleting checkpoints")
    for batch_idx in range(ceildiv(len(prompts), save_every) ):
        os.remove(save_path + f"/{file_spec}check{batch_idx}_all_hidden_states.pt")
        
    return all_hidden_states, all_generations, all_tokens, all_mem_status
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
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
    parser.add_argument(
        '--seed', 
        type=int,
        default=0,
    )


    args = parser.parse_args()
    
    print(args)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    mw = ModelWrapper(model, tokenizer)
    
    if 'pythia-evals' in args.dataset: 
        if "deduped" in args.model_name:
            #model_name looks like: EleutherAI/pythia-12B-deduped
            dataset_name = "deduped." + args.model_name.split("-")[-2]
        else:
            #model_name looks like: EleutherAI/pythia-12B
            dataset_name = "duped." + args.model_name.split("-")[-1]
        
        if args.dataset == 'pythia-evals-12b': 
            dataset_name = 'duped.12b'
        
        mem_data = load_dataset('EleutherAI/pythia-memorized-evals')[dataset_name]

        toks = [seq for seq in mem_data[:args.N_PROMPTS]['tokens']]
        for i in range(len(toks)): 
            left = 64 - len(toks[i])
            assert left == 0, "Need to pad left"
            toks[i] = [tokenizer.pad_token_id] * left + toks[i] # pad left as suggested above
        toks = torch.tensor(toks)
        prompts = [toks_to_string(tokenizer, seq) for seq in toks]
        file_spec = 'mem_'
    
    elif args.dataset == 'pile': 
        prompts = gen_pile_data(args.N_PROMPTS, tokenizer, min_n_toks = 64)
        toks = tokenizer(prompts, return_tensors = 'pt', padding = True, max_length = 64, truncation = True)['input_ids']
        file_spec = 'pile_'        

    elif args.dataset == 'pile-test': 
        prompts = gen_pile_data(args.N_PROMPTS, tokenizer, min_n_toks = 64, split='validation')
        toks = tokenizer(prompts, return_tensors = 'pt', padding = True, max_length = 64, truncation = True)['input_ids']
        file_spec = 'pile_test_'
        
    tok_idxs =  (7 * np.arange(10)).tolist() #every 5th token
    tok_idxs[-1]= tok_idxs[-1] - 1 #goes from 63 to 62
    print(tok_idxs)

    hidden_states, generations, gen_tokens, mem_status = get_memmed_activations(mw,
                                                                            toks, 
                                                                            args.save_path,
                                                                            save_every = args.save_every,
                                                                            # check_if_memmed = args.check_if_memmed,
                                                                            N_TOKS = args.N_TOKS,
                                                                            layers = args.layers,
                                                                            tok_idxs = tok_idxs,
                                                                            return_prompt_acts = args.return_prompt_acts,
                                                                            logging = args.logging,
                                                                            file_spec = file_spec)
    
    if args.act_types: 
        acts_dict = get_memmed_activations_from_pregenned(mw,
                                                        gen_tokens,
                                                        args.save_path,
                                                        act_types = args.act_types,
                                                        save_every = args.save_every,
                                                        layers = args.layers,
                                                        tok_idxs = tok_idxs,
                                                        logging = args.logging,
                                                        file_spec = file_spec + "attn_mlp_")

    print("Done")