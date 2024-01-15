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

from utils import untuple

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
    acts = torch.cat([torch.cat(out.hidden_states[i], dim = 0) for i in range(1, N_TOKS)], dim = 1)  #1, N_TOKS bc the first index is all previous tokens
    #shape: n_layers + 1, N_TOKS - 1, d_M
    #n_layers + 1 bc of embedding, N_TOKS - 1 bc of how max_new_tokens works
    
    if return_prompt_acts:
        prompt_acts = torch.cat(out.hidden_states[0], dim = 0)
        acts = torch.cat([prompt_acts, acts], dim = 1)
    
    acts = acts.cpu()
    
    if tok_idxs is not None:
        acts = acts[:, tok_idxs, :]
    acts = acts[layers, :, :]
    return acts

def ceildiv(a, b):
    #https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)

def get_memmed_activations(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: Union[List[str],List[int]],
                    save_path: str,
                    save_every: int = 100, 
                    check_if_memmed: bool = True,
                    N_TOKS: int = 32,
                    layers: List = None,
                    tok_idxs: List = None,
                    return_prompt_acts: bool = False,
                    logging: bool = False,
                    file_spec: str = "",
                    **generation_kwargs):
    
    if not os.path.exists(save_path):
        os.makedirs(args.save_path)
    
    if layers is None:
        layers = range(1, model.config.num_hidden_layers + 1)
        
    all_generations = []
    all_hidden_states = []
    all_mem_status = []
    
    for batch_idx in range(ceildiv(len(prompts), save_every) ):
        if logging:
            start_time = time.time()
            print(f"Batch {batch_idx + 1} of {len(prompts) // save_every}")
            
        batch = prompts[batch_idx * save_every : (batch_idx + 1) * save_every]
        
        batch_hidden_states = []
        
        for index_in_batch, s in enumerate(batch):
            if isinstance(s[0], str):
                toks = tokenizer(s, return_tensors="pt", max_length = N_TOKS * 2, truncation = True)
                input_ids = toks.input_ids.to(model.device)[:, :N_TOKS] #shape [1, N_TOKS]
                toks = toks.input_ids[0].numpy().tolist()
            else:
                toks = s
                input_ids = torch.tensor(toks[:N_TOKS], device = model.device).unsqueeze(0)
            
            out = model.generate(
                        input_ids=input_ids,
                        top_p = 1.0,
                        max_new_tokens=N_TOKS,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                        **generation_kwargs,
                        )
            
            out_toks = out.sequences[0].cpu().numpy().tolist()
            
            if len(out_toks) != 64:
                print("Generation failed, skipping this one.")
                print(tokenizer.decode(out_toks, skip_special_tokens=True))
                print(index_in_batch)
                print()
                continue
            
            generation = tokenizer.decode(out_toks, skip_special_tokens=True)
            
            if check_if_memmed:
                all_mem_status.append(compare_token_lists(toks[:2*N_TOKS], out_toks))

            acts = slice_acts(out, N_TOKS, return_prompt_acts, layers, tok_idxs)
            
            batch_hidden_states.append(acts)
            
            all_generations.append(generation)
            
        if logging:
            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time
            print(f"Time elapsed for this batch: {elapsed_time:.2f} seconds")
            print(f"Generations:")
            for g, m in zip(all_generations[-len(batch_hidden_states):], all_mem_status[-len(batch_hidden_states):]):
                print(g)
                print(m)
                print()
            print(f"Num Hidden States Genned: {len(batch_hidden_states)}")
            print(f"Shape of One Hidden State: {batch_hidden_states[0].shape}")
            print()
            
        all_hidden_states.extend(batch_hidden_states)
        
        del batch_hidden_states
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save all_hidden_states
        torch.save(all_hidden_states, save_path + f"/{file_spec}all_hidden_states.pt")

        # Save all_generations
        with open(save_path + f"/{file_spec}all_generations.pkl", "wb") as f:
            pickle.dump(all_generations, f)

        # Save all_mem_status
        with open(save_path + f"/{file_spec}all_mem_status.pkl", "wb") as f:
            pickle.dump(all_mem_status, f)
    
    all_hidden_states = torch.stack(all_hidden_states, dim = 0)
    torch.save(all_hidden_states, save_path + f"/{file_spec}all_hidden_states.pt")
    
    return all_hidden_states, all_generations, all_mem_status
    
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
        "--check_if_memmed",
        action="store_true",
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
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if "deduped" in args.model_name:
        #model_name looks like: EleutherAI/pythia-12B-deduped
        dataset_name = "deduped." + args.model_name.split("-")[-2]
    else:
        #model_name looks like: EleutherAI/pythia-12B
        dataset_name = "duped." + args.model_name.split("-")[-1]
    
    mem_data = load_dataset('EleutherAI/pythia-memorized-evals')[dataset_name]

    mem_data_toks = [seq for seq in mem_data[:args.N_PROMPTS]['tokens']]
    
    # mem_data_prompts = [toks_to_string(tokenizer, seq) for seq in mem_data_toks]
    
    pile_prompts = gen_pile_data(args.N_PROMPTS, tokenizer, min_n_toks = 64)
    print(len(pile_prompts))
    print(len(mem_data_toks))
    
    tok_idxs =  (7 * np.arange(10)).tolist() #every 5th token
    tok_idxs[-1]= tok_idxs[-1] - 1 #goes from 63 to 62
    print(tok_idxs)
    mem_hidden_states, mem_generations, mem_mem_status = get_memmed_activations(model, 
                                                                                tokenizer, 
                                                                                mem_data_toks, 
                                                                                args.save_path,
                                                                                save_every = args.save_every,
                                                                                check_if_memmed = args.check_if_memmed,
                                                                                N_TOKS = args.N_TOKS,
                                                                                layers = args.layers,
                                                                                tok_idxs = tok_idxs,
                                                                                return_prompt_acts = args.return_prompt_acts,
                                                                                logging = args.logging,
                                                                                file_spec = "mem_")
    
    pile_hidden_states, pile_generations, pile_mem_status = get_memmed_activations(model,
                                                                                    tokenizer, 
                                                                                    pile_prompts, 
                                                                                    args.save_path,
                                                                                    save_every = args.save_every,
                                                                                    check_if_memmed = args.check_if_memmed,
                                                                                    N_TOKS = args.N_TOKS,
                                                                                    layers = args.layers,
                                                                                    tok_idxs = tok_idxs,
                                                                                    return_prompt_acts = args.return_prompt_acts,
                                                                                    logging = args.logging,
                                                                                    file_spec = "pile_")
    
    
    