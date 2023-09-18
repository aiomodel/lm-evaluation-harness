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


def _select_cont_toks(logits, contlen=None, inplen=None):
    assert (
        contlen and inplen
    ), "Must pass input len and cont. len to select scored logits for causal LM"
    # discard right-padding.
    # also discard the input/context tokens. we'll only score continuations.
    logits = logits[inplen - contlen : inplen]
    return logits


def forward_scores(local_rank, reward_model_path, shared_list, max_size):
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
        data = json.load(open(f"/mnt/lm-evaluation-harness-refactor/data/{local_rank}.json"))[str(local_rank)]
        results = {}

        # pad size
        for i in range(len(data.keys()), max_size):
            data[-i] = data[list(data.keys())[0]]

        for index, (batched_inps, chunk, inplens, cont_toks_list, padding_len_inp) in tqdm.tqdm(data.items()):
            multi_logits = F.log_softmax(
                score_model(input_ids=torch.tensor(batched_inps).cuda()).logits, dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            result = []
            for (cache_key, _, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = _select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(
                    cont_toks, dtype=torch.long
                ).cuda().unsqueeze(
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
    max_size = 0
    for i in range(torch.cuda.device_count()):
        data = json.load(open(f"/mnt/lm-evaluation-harness-refactor/data/{i}.json"))[str(i)]
        print(len(data.keys()), max_size)
        max_size = max(len(data.keys()), max_size)
    mp.spawn(forward_scores, nprocs=torch.cuda.device_count(), 
             args=(reward_model_path, shared_list, max_size))
    shared_list = sorted(shared_list, key=lambda x: x[0])
    
    # save results
    for local_rank, results in shared_list:
        json.dump(results, open(f"/mnt/lm-evaluation-harness-refactor/data/{local_rank}-results.json", "w"))


if __name__ == "__main__":
    app()