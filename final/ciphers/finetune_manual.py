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
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    # dataset = dataset.remove_columns(['text', 'attention_mask'])
    # dataset.set_format(type='torch', columns=['input_ids', 'labels'])
    dataset = dataset.train_test_split(test_size=0.1)

    ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')
        
        # train
        model.train()
        total_loss = 0
        for i in tqdm(range(0, len(dataset['train']), batch_size)):
            inputs = dataset['train']['input_ids'][i:i+batch_size]
            inputs = torch.tensor(inputs).to(model.device)
            logits = model(inputs).logits
            logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            labels = inputs[:, 1:].contiguous().view(-1)

            # replace all padding tokens with -100 before calculating loss
            labels[labels == tokenizer.pad_token_id] = -100

            loss = ce(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        print(f'Training Loss: {total_loss / (len(dataset["train"]) / batch_size)}')

        # eval
        model.eval()
        total_loss = 0
        for i in tqdm(range(0, len(dataset['test']), batch_size)):
            inputs = dataset['test']['input_ids'][i:i+batch_size]
            inputs = torch.tensor(inputs).to(model.device)
            logits = model(inputs).logits
            logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            labels = inputs[:, 1:].contiguous().view(-1)

            # replace all padding tokens with -100 before calculating loss
            labels[labels == tokenizer.pad_token_id] = -100

            loss = ce(logits, labels)
            total_loss += loss.item()
        
        print(f'Validation Loss: {total_loss / (len(dataset["test"]) / batch_size)}')
        model.save_pretrained(os.path.join(args.path, args.output_name + f'_epoch_{i}'))

    model.save_pretrained(os.path.join(args.path, args.output_name + '_final'))

    # remove prev models
    for i in range(num_epochs):
        os.remove(os.path.join(args.path, args.output_name + f'_epoch_{i}'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/ubuntu/gld/train-data-probes/data/70m/ciphers', help='path to data')
    parser.add_argument('--dataset_name', type=str, default='rotated_3', help='name of dataset')
    parser.add_argument('--output_name', type=str, default='rotated_3_model', help='name of output')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m', help='name of model')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    args = parser.parse_args()

    main(args)