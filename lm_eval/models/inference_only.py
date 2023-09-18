"""
policy = LLM()
results = policy.generate(prompts, n)
reward_model = RewardModel()

rewards = reward_model(prompts+results)
best_idx, median_idx, best_reward, median_reward

"""

import os
import json
import math
from pathlib import Path
import numpy as np
import tqdm
import functools
import typer
from typing import List, Tuple, Dict
import torch.nn.functional as F

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import datasets
import transformers
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from vllm import LLM, SamplingParams
import ray

# FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import  ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import  transformer_auto_wrap_policy
from time import sleep
from time import perf_counter

app = typer.Typer()

def load_reward_model(
    reward_path:str,
):
    """
    load reward model to cuda device
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        reward_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    return model


def forward_scores(local_rank, reward_model_path, shared_list):
    try:
        world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except:
        world_size = torch.cuda.device_count() # TODO: only support single node

    dist.init_process_group(
    	backend='nccl',                                         
    	world_size=world_size,
    	rank=local_rank, # TODO: only support single node
    )
    print(f"global_rank={dist.get_rank()}/{world_size}, local_rank={local_rank}/{torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)

    with torch.inference_mode():
        # TODO: when world_size>4, need to sleep half of rank to avoid cpu oom
        if world_size > 4:
            if local_rank % 2 == 0:
                sleep_time = 45 * 2
                print(f"sleep {sleep_time} seconds at rank {local_rank}")
                sleep(sleep_time)
        score_model = load_reward_model(reward_model_path)
        # PyTorch FSDP!
        print("doing fsdp...")
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls={LlamaDecoderLayer},
        )
        score_model = FSDP(
            score_model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=True),
            auto_wrap_policy=auto_wrap_policy,
            device_id=local_rank
        )
        score_model = score_model.eval()
        
        # load data
        data = json.load(open(f"../../data/data.json"))
        results = {}

        # split data
        lengths = len(data)
        if lengths % world_size != 0:
            lengths_per_size = lengths // world_size + 1
        else:
            lengths_per_size = lengths // world_size
        splitted_data = {}
        item_count = 0
        for k, v in data.items():
            if item_count >= lengths_per_size * local_rank and item_count < lengths_per_size * (local_rank + 1):
                splitted_data[k] = v
            item_count += 1
        # pad size
        if len(splitted_data) < lengths_per_size:
            print(lengths_per_size, len(splitted_data), len(data), local_rank)
            pad_item = splitted_data[list(splitted_data.keys())[0]]
            for i in range(len(splitted_data.keys()), lengths_per_size):
                splitted_data[-i] = pad_item
        data = splitted_data

        for index, (batched_inps, chunk, inplens, cont_toks_list) in tqdm.tqdm(data.items()):
            multi_logits = F.log_softmax(
                score_model(input_ids=torch.tensor(batched_inps).cuda()).logits, dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            result = []
            for (cache_key, _, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[inplen - contlen : inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).cuda().unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))
                result.append([cache_key, answer])

            if int(index) < 0:
                # pad part
                pass
            else:
                results[index] = [batched_inps, result]

        shared_list.append([local_rank, results])


@app.command()
def main(
    reward_model_path:str,
):

    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    print(f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    manager = mp.Manager()
    shared_list = manager.list()
    
    mp.spawn(forward_scores, nprocs=torch.cuda.device_count(), 
             args=(reward_model_path, shared_list))
    shared_list = sorted(shared_list, key=lambda x: x[0])

    results = {}
    # save results
    for local_results in shared_list:
        local_rank, local_results = local_results
        for k, v in local_results.items():
            results[k] = v
    data = json.load(open(f"../../data/data.json"))
    assert len(results) == len(data), f"{len(results)} v.s. {len(data)}"
    json.dump(results, open(f"../../data/results.json", "w"))


if __name__ == "__main__":
    app()