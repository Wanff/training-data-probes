import re
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import Levenshtein

def untuple(x):
    return x[0] if isinstance(x, tuple) else x
    
def tpr_at_fpr(probs, labels, target_fpr, left=0.5, right=1.0, max_steps=1000, thresh_tol=1e-6): 
    """
    Calculates the true positive rate at a given false positive rate. 
    Does up to max_steps steps of binary search, returns the best guess 
    that yields a false positive rate less than target_fpr
    
    probs: (n_examples, ) just prob on positive class
    labels: (n_examples, ) 0 or 1
    """
    assert len(probs) == len(labels)
    assert probs.shape == labels.shape
    assert probs.shape[0] > 0

    for _ in range(max_steps):
        mid = (left + right) / 2
        
        # calc fpr 
        preds = (probs > mid).astype(int)
        fp = np.logical_and(preds == 1, labels == 0).sum()
        tn = (labels == 0).sum()
        fpr = fp / tn if tn > 0 else 0

        if abs(fpr - target_fpr) < thresh_tol: 
            right = mid
            break
        elif fpr > target_fpr:
            left = mid
        else:
            right = mid
    
    # use right as threshold to ensure fpr <= target_fpr
    preds = (probs > right).astype(int)
    tp = np.logical_and(preds == 1, labels == 1).sum()
    fn = (labels == 1).sum()
    return tp / fn if fn > 0 else 0


sim_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
def sim_scores(outputs, targets):
    semantic_scores_gen = []
    for target, output in zip(targets, outputs):
        embedding1 = sim_model.encode(target, convert_to_tensor=True)
        embedding2 = sim_model.encode(output, convert_to_tensor=True)
        cosine_sim_gen = util.pytorch_cos_sim(embedding1, embedding2)
        similarity_value_gen = cosine_sim_gen.item()
        semantic_scores_gen.append(similarity_value_gen)
    
    return semantic_scores_gen 

def char_by_char_similarity(outputs, targets):
    similarities = []
    for o, t in zip(outputs, targets):
        o = re.sub(r'\s', '', o)
        t = re.sub(r'\s', '', t)

        o = o.lower()
        t = t.lower()

        # remove '<|endoftext|>'
        o = o.replace('<|endoftext|>', '')
        t = t.replace('<|endoftext|>', '')

        max_len = max(len(o), len(t))
        matches = [c1 == c2 for c1, c2 in zip(o, t)]
        
        similarities.append(sum(matches)/max_len if max_len > 0 else 0)
    return similarities

def compare_token_lists(genned_toks, ground_toks):
    if len(ground_toks) < len(genned_toks):
        num_same_tokens = sum(1 for token1, token2 in zip(ground_toks, genned_toks[:len(ground_toks)]) if token1 == token2)
        percent_same_tokens = (num_same_tokens / len(ground_toks)) 
        return percent_same_tokens
    elif len(ground_toks) > len(genned_toks):
        print("Ground truth is longer than generated text. This should not happen.")
        print(len(ground_toks), len(genned_toks))
        return 0
    else:
        num_same_tokens = sum(1 for token1, token2 in zip(ground_toks, genned_toks) if token1 == token2)
        percent_same_tokens = (num_same_tokens / len(ground_toks)) 
        
        return percent_same_tokens

def tok_by_tok_similarity(outputs, targets, tokenizer = None):
    if isinstance(outputs[0], str):
        assert tokenizer is not None
        outputs = tokenizer(outputs, return_tensors = 'pt',padding = False, truncation = True, max_length = 64)['input_ids']
        targetse = tokenizer(targets, return_tensors = 'pt',padding = False, truncation = True, max_length = 64)['input_ids']

    return [compare_token_lists(t, o) for t, o in zip(outputs, targets)]

def levenshtein_distance(outputs, targets):
    diss = []
    for o, t in zip(outputs, targets):
        max_len = max(len(o), len(t))
        diss.append((max_len - Levenshtein.distance(o, t)) / max_len)
    return diss

def extract_quote_completion(s):
    s = s.replace(";",",").split(".")[0].split("\n")[0]
    return s.strip().lower()


def eval_completions(outputs, targets, sim_types = ['char', 'tok', 'lev', 'sem'], tokenizer = None, return_mean = True):
    return_dict = {}
    if 'char' in sim_types:
        cbc_sims = char_by_char_similarity(outputs, targets)
        return_dict['char_by_char_similarity'] = cbc_sims
    
    if 'tok' in sim_types:
        tbt_sims = tok_by_tok_similarity(outputs, targets, tokenizer = tokenizer)
        return_dict['tok_by_tok_similarity'] = tbt_sims
    
    if 'lev' in sim_types:
        lev_diss = levenshtein_distance(outputs, targets)
        return_dict['lev_distance'] = lev_diss
    
    if 'sem' in sim_types:
        sem_sims = sim_scores(outputs, targets)
        return_dict['sem_similarity'] = sem_sims
    
    if return_mean:
        return {k: np.mean(v) for k, v in return_dict.items()}
   
    return return_dict