from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, AutoModelForCausalLM
import evaluate
import torch
import os
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import argparse

def main(args): 

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(os.path.join(args.path, args.dataset_name)) # should have a train and test split with 'text' column
    model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, device_map='auto')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    dataset = dataset.train_test_split(test_size=0.1)

    targs = TrainingArguments(
        output_dir = os.path.join(args.path, args.output_name),
        evaluation_strategy = 'epoch',
        logging_strategy = 'epoch',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        report_to='none',
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=targs,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'], 
    )

    trainer.train()
    model.save_pretrained(os.path.join(args.path, args.output_name + '_final'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/ubuntu/gld/train-data-probes/data/1b/ciphers', help='path to data')
    parser.add_argument('--dataset_name', type=str, default='rotated_3', help='name of dataset')
    parser.add_argument('--output_name', type=str, default='rotated_3_model', help='name of output')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-1b', help='name of model')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    args = parser.parse_args()

    main(args)