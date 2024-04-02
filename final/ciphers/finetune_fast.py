"""
Finetune the Llama model on the answer behavior dataset.

Usage:
python finetune_llama.py --behavior sycophancy --direction pos 
"""

import argparse
import json
from typing import Literal, Optional
import torch as t
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import numpy as np
from datasets import load_from_disk
import os
from dotenv import load_dotenv
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
load_dotenv()

t.autograd.set_detect_anomaly(True)

NUM_PROCESSES = 1

class TextDataset(Dataset):
    def __init__(self, data): 
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data['input_ids'][idx]

def finetune(
    rank: int,
    world_size: int,
    path: str,
    dataset_name: str,
    output_name: str,
    model_name: str,
    n_epochs: int,
    lr: float,
    batch_size: int,
): 
    # Initialize distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Barrier to ensure all processes have initialized
    dist.barrier()
    # Device corresponding to current process
    DEVICE = t.device(f"cuda:{rank}") if t.cuda.is_available() else "cpu"
    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    )
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    params = ddp_model.parameters()
    # Set up the optimizer
    optimizer = ZeroRedundancyOptimizer(params, optimizer_class=t.optim.SGD, lr=lr)

    # get dataset

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    dataset = load_from_disk(os.path.join(path, dataset_name))
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(['text', 'attention_mask'])
    dataset.set_format(type='torch', columns=['input_ids'])
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = TextDataset(dataset['train'])
    test_dataset = TextDataset(dataset['test'])

    # Setup the DataLoader with DistributedSampler
    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    dataloader = DataLoader(train_dataset, batch_size=1, sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Run the training loop
    loss_fn = t.nn.CrossEntropyLoss()
    # Time the training
    start_time = datetime.datetime.now()
    for epoch in range(n_epochs):
        print_every = len(dataloader) // 10
        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        avg_loss = 0
        n_batches = 0
        for i, input_ids in enumerate(dataloader):
            
            input_ids = input_ids.to(DEVICE)
            logits = ddp_model(input_ids).logits
            logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            labels = input_ids[:, 1:].contiguous().view(-1)

            # replace all padding tokens with -100 before calculating loss
            labels[labels == tokenizer.pad_token_id] = -100

            loss = loss_fn(logits, labels)
            avg_loss += loss.item()
            n_batches += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if i % print_every == 0:
                print(
                    f"Rank: {rank} | Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss / n_batches}"
                )
                avg_loss = 0
                n_batches = 0
        
        # eval
        ddp_model.eval()
        with t.no_grad():
            total_loss = 0
            n_batches = 0
            for i, input_ids in enumerate(test_dataloader): 
                input_ids = input_ids.to(DEVICE)
                logits = ddp_model(input_ids).logits
                logits = logits[:, :-1].view(-1, logits.size(-1))
                labels = input_ids[:, 1:].view(-1)

                # replace all padding tokens with -100 before calculating loss
                # labels[labels == tokenizer.pad_token_id] = -100

                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                n_batches += 1
            print(f"Rank: {rank} | Epoch {epoch + 1}/{n_epochs} | Test Loss: {total_loss / n_batches}")
        
        # Save the model after each epoch
        t.save(ddp_model.module.state_dict(), os.path.join(path, output_name + f'_epoch_{epoch}'))

        
    end_time = datetime.datetime.now()
    # Finalize the training
    dist.barrier()
    if rank == 0:
        
        # Save the model after training completes
        t.save(ddp_model.module.state_dict(), os.path.join(path, output_name + '_final'))
        print(f"Training completed in {end_time - start_time}")
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    # Ensure the CUDA devices are available and spawn the training processes
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/ubuntu/gld/train-data-probes/data/70m/ciphers', help='path to data')
    parser.add_argument('--dataset_name', type=str, default='rotated_3', help='name of dataset')
    parser.add_argument('--output_name', type=str, default='rotated_3_model', help='name of output')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m', help='name of model')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    set_seed(args.seed)

    try:
        if t.cuda.is_available() and t.cuda.device_count() >= NUM_PROCESSES:
            # Number of GPUs or processes you want to run
            world_size = NUM_PROCESSES
            mp.spawn(
                finetune,
                args=(world_size, args.path, args.dataset_name, args.output_name, args.model_name, args.num_epochs, args.lr, args.batch_size),
                nprocs=world_size,
            )
        else:
            raise EnvironmentError(
                f"Ensure that you have {NUM_PROCESSES} GPUs available."
            )
    except Exception as e:
        print(f"An error occurred: {e}")