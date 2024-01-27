from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
import evaluate
import torch
import os
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import gc
import sys
sys.path.append('..')
from utils import tpr_at_fpr
from peft import PeftModel, LoraConfig, TaskType, PeftConfig

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
set_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.set_device(0)

model_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
orig_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
tokenizer.pad_token = tokenizer.eos_token

path = '../../../gld/train-data-probes/data/12b'
dataset = load_from_disk(os.path.join(path, 'split_hf_token_dataset_vary_len_v2'))
generalization_datasets = load_from_disk(os.path.join(path, 'generalization_datasets_v2'))
id2label = {0: 'neg', 1: 'pos'}
label2id = {'neg': 0, 'pos': 1}
peft_config = PeftConfig.from_pretrained(os.path.join(path, 'finetuning/final-llama-on-pythia-2/checkpoint-25'))
model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=2, id2label=id2label, label2id=label2id, low_cpu_mem_usage=True, device_map='auto')
model.config.pad_token_id = model.config.eos_token_id
model = PeftModel.from_pretrained(model, os.path.join(path, 'finetuning/final-llama-on-pythia-2/checkpoint-25'))

# pad datasets

def preprocess(example): 
    return {'input_ids': tokenizer(orig_tokenizer.decode(example['input_ids']), return_tensors='pt')['input_ids'][0]}

dataset['train'] = dataset['train'].map(preprocess)
dataset['val'] = dataset['val'].map(preprocess)
dataset['test'] = dataset['test'].map(preprocess)

max_train_len = max([len(x['input_ids']) for x in dataset['train']])
max_val_len = max([len(x['input_ids']) for x in dataset['val']])
max_test_len = max([len(x['input_ids']) for x in dataset['test']])

def preprocess(examples, max_len): 
    curr_len = len(examples['input_ids'])
    if curr_len < max_len:
        examples['input_ids'] = examples['input_ids'] + [tokenizer.pad_token_id]*(max_len-curr_len)
    return examples

dataset['train'] = dataset['train'].map(lambda x: preprocess(x, max_train_len))
dataset['val'] = dataset['val'].map(lambda x: preprocess(x, max_val_len))
dataset['test'] = dataset['test'].map(lambda x: preprocess(x, max_test_len))

# pad datasets

def preprocess(example): 
    return {'input_ids': tokenizer(orig_tokenizer.decode(example['input_ids']), return_tensors='pt')['input_ids'][0]}

generalization_datasets['prefix'] = generalization_datasets['prefix'].map(preprocess)
generalization_datasets['fuzzy_pos'] = generalization_datasets['fuzzy_pos'].map(preprocess)

max_prefix_len = max([len(x['input_ids']) for x in generalization_datasets['prefix']])
max_fuzzy_pos_len = max([len(x['input_ids']) for x in generalization_datasets['fuzzy_pos']])

def preprocess(examples, max_len): 
    curr_len = len(examples['input_ids'])
    if curr_len < max_len:
        examples['input_ids'] = examples['input_ids'] + [tokenizer.pad_token_id]*(max_len-curr_len)
    return examples

generalization_datasets['prefix'] = generalization_datasets['prefix'].map(lambda x: preprocess(x, max_prefix_len))
generalization_datasets['fuzzy_pos'] = generalization_datasets['fuzzy_pos'].map(lambda x: preprocess(x, max_fuzzy_pos_len))


train_lens = [len(x['input_ids']) for x in dataset['train']]
val_lens = [len(x['input_ids']) for x in dataset['val']]
test_lens = [len(x['input_ids']) for x in dataset['test']]

print(set(train_lens), set(val_lens), set(test_lens))

for gen_dataset in generalization_datasets.values(): 
    gen_lens = [len(x['input_ids']) for x in gen_dataset]
    print(set(gen_lens))

loss_fn = torch.nn.CrossEntropyLoss()
data = {'val_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'AUC': [], 'tpr_01': [], 'tpr_001': [], 'tpr_0001': []}

def compute_metrics(eval_pred, single_class=False):
    
    logits, labels = eval_pred
    logits = np.array(logits) # (n_examples, n_classes)
    labels = np.array(labels) # (n_examples)

    preds = np.argmax(logits, axis=1)
    normalized_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    positive_prob = normalized_prob[:, 1]
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    auc = 0
    if not single_class: 
        auc = roc_auc_score(labels, positive_prob)
    
    loss = loss_fn(torch.tensor(logits), torch.tensor(labels))

    # calculate tpr at fpr = 0.01, 0.001, and 0.0001
    tpr_01 = tpr_at_fpr(positive_prob, labels, 0.01)
    tpr_001 = tpr_at_fpr(positive_prob, labels, 0.001)
    tpr_0001 = tpr_at_fpr(positive_prob, labels, 0.0001)

    # log
    data['val_loss'].append(loss.item())
    data['accuracy'].append(acc)
    data['precision'].append(precision)
    data['recall'].append(recall)
    data['f1'].append(f1)
    if not single_class:
        data['AUC'].append(auc)
    data['tpr_01'].append(tpr_01)
    data['tpr_001'].append(tpr_001)
    data['tpr_0001'].append(tpr_0001)

    return {'val_loss': loss.item(), 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'AUC': auc, 'tpr_01': tpr_01, 'tpr_001': tpr_001, 'tpr_0001': tpr_0001}

targs = TrainingArguments(
            output_dir = os.path.join(path, 'finetuning/final'),
            evaluation_strategy = 'epoch',
            eval_steps=1, 
            logging_strategy = 'epoch',
            logging_steps=1,
            save_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=1,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model='val_loss',
            weight_decay=0.01,
            report_to='none',
            seed=seed,
            eval_accumulation_steps=1,
        )

# get random 100 sized subset of dataste['val']
rd = dataset['val'].select(np.random.choice(len(dataset['val']), 100, replace=False))
trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=targs,
            train_dataset=dataset['train'],
            eval_dataset=rd,
            compute_metrics=compute_metrics,
        )

print(trainer.evaluate())
print(data)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=targs,
    train_dataset=dataset['train'],
    eval_dataset=generalization_datasets['fuzzy_pos'], 
    compute_metrics=lambda x: compute_metrics(x, single_class=True),
)

print(trainer.evaluate())
print(data)

targs = TrainingArguments(
            output_dir = os.path.join(path, 'finetuning/final'),
            evaluation_strategy = 'epoch',
            eval_steps=1, 
            logging_strategy = 'epoch',
            logging_steps=1,
            save_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=1,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model='val_loss',
            weight_decay=0.01,
            report_to='none',
            seed=seed,
            eval_accumulation_steps=1,
        )

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=targs,
    train_dataset=dataset['train'],
    eval_dataset=generalization_datasets['prefix'].select(range(len(generalization_datasets['prefix'])//2)),
    compute_metrics=compute_metrics,
)

print(trainer.evaluate())
print(data)

