import re
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import Levenshtein
import datasets

#from https://github.com/saprmarks/geometry-of-truth/blob/main/probes.py
import torch as t
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class Probe:
    def __init__(self, d_m):
        self.d_m = d_m

    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def predict_proba(self, x):
        raise NotImplementedError

    def get_probe_accuracy(self, X_test, y_test, device = "cpu"):
        if device != "cpu":
            X_test = X_test.to(device)
            y_test = y_test.to(device)
        preds = self.predict(X_test)

        accuracy = (preds == y_test).float().mean().item()
        return accuracy

    def get_probe_auc(self, X_test, y_test, device = "cpu"):
        if device != "cpu":
            X_test = X_test.to(device)
            y_test = y_test.to(device)
        preds = self.predict_proba(X_test)
        auc = roc_auc_score(y_test.detach().cpu().numpy(), preds.detach().cpu().numpy())
        return auc

class MLP(t.nn.Module, Probe):
    def __init__(self, d_in, d_out, d_hidden, n_hidden, use_bias=False):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, d_hidden, bias=use_bias),
            t.nn.ReLU(),
            *[t.nn.Sequential(
                t.nn.Linear(d_hidden, d_hidden, bias=use_bias),
                t.nn.ReLU()
            ) for _ in range(n_hidden)],
            t.nn.Linear(d_hidden, d_out, bias=use_bias),
            t.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def predict(self, x):
        return self(x).round()
    
    def predict_proba(self, x):
        return self(x)
    
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, labels = acts.to(device), labels.to(device)
        probe = MLP(acts.shape[-1], 1, 32, 2).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in tqdm(range(epochs)):
            opt.zero_grad()
            loss = t.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()
        
        return probe

class MoEProbe(t.nn.Module, Probe):
    def __init__(self, d_m, probes, probe_type = "sk", use_bias=False, device = "cuda"):
        super().__init__()
        self.d_m = d_m
        self.probe_type = probe_type
        self.probes = probes
        self.n_experts = len(probes)
        
        self.gater = t.nn.Sequential(
            t.nn.Linear(d_m * self.n_experts, self.n_experts, bias=use_bias, dtype=t.float32).to(device),
            # t.nn.Softmax(dim = -1)
        )
   
    def forward(self, x):
        assert x.dtype == t.float32
        assert x.device == self.gater[0].weight.device
        
        with t.no_grad():
            if self.probe_type == "sk":
                probe_preds = t.stack([ t.tensor(probe.predict_proba(x[:, i*self.d_m:(i+1)*self.d_m].cpu().numpy())[:, 1]) for i, probe in enumerate(self.probes)], dim=-1).to("cuda")
            else:
                probe_preds = t.stack([probe(x[:, i*self.d_m:(i+1)*self.d_m]) for i, probe in enumerate(self.probes)], dim=-1)
        
        gater_pred = self.gater(x)
        out = (probe_preds * gater_pred).sum(-1)
        return t.sigmoid(out)

    def predict(self, x):
        return self(x).round()
    
    def predict_proba(self, x):
        return self(x)
    
    def train(self, acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, labels = acts.to(device), labels.to(device)
        
        opt = t.optim.AdamW(self.gater.parameters(), lr=lr, weight_decay=weight_decay)
        
        losses = []
        for _ in tqdm(range(epochs)):
            opt.zero_grad()
            loss = t.nn.BCELoss()(self(acts), labels)
            loss.backward()
            losses.append(loss.item())
            opt.step()
        
        return loss
    
class LRProbe(t.nn.Module, Probe):
    def __init__(self, d_in, use_bias=False):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=use_bias),
            t.nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def predict(self, x, iid=None):
        return self(x).round()
    
    def predict_proba(self, x, iid=None):
        if x.device != "cuda":
            x = x.to("cuda")
        return self(x)
    
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, use_bias=False, device='cpu'):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1], use_bias = use_bias).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        
        for _ in range(epochs):
            opt.zero_grad()
            loss = t.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()
        
        return probe
    
    def from_weights(weights, bias = None, device = "cpu"):
        probe = LRProbe(weights.shape[1], use_bias = bias is not None)
        probe.net[0].weight.data = weights.float().to(device)
        
        if bias is not None:
            probe.net[0].bias.data = bias.float().to(device)
            
        return probe
    
    @property
    def direction(self):
        return self.net[0].weight.data[0]

class MMProbe(t.nn.Module, Probe):
    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        if inv is None:
            self.inv = t.nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
        else:
            self.inv = t.nn.Parameter(inv, requires_grad=False)

    def forward(self, x, iid=False):
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            return t.nn.Sigmoid()(x @ self.direction)

    def predict(self, x, iid=False):
        return self(x, iid=iid).round()

    def predict_proba(self, x, iid=False):
        return self(x, iid=iid)
    
    def from_data(acts, labels, atol=1e-3, device='cpu'):
        acts, labels
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]
        
        probe = MMProbe(direction, covariance=covariance).to(device)

        return probe

def ccs_loss(probe, acts, neg_acts):
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = t.min(t.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return t.mean(consistency_losses + confidence_losses)


class CCSProbe(t.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )
    
    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)
    
    def pred(self, acts, iid=None):
        return self(acts).round()
    
    def from_data(acts, neg_acts, labels=None, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, neg_acts = acts.to(device), neg_acts.to(device)
        probe = CCSProbe(acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        if labels is not None: # flip direction if needed
            acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1
        
        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]

def untuple(x):
    return x[0] if isinstance(x, tuple) else x
    
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
        if tokenizer.padding_side == "left":
            print("Are you sure you want padding_side == left?")
            
        outputs = tokenizer(outputs, return_tensors = 'pt',padding = True, truncation = True)['input_ids']
        targets = tokenizer(targets, return_tensors = 'pt',padding = True)['input_ids']
        max_length = max(outputs.shape[1], targets.shape[1])

        if outputs.shape[1] < max_length:
            outputs = torch.cat([outputs, torch.zeros(outputs.shape[0], max_length - outputs.shape[1], dtype = torch.long)], dim = 1)
        if targets.shape[1] < max_length:
            targets = torch.cat([targets, torch.zeros(targets.shape[0], max_length - targets.shape[1], dtype = torch.long)], dim = 1)
            
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