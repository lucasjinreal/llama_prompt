from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA, LLaMA2


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA2(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,  # 16 for 13B, 4 for 30B and 65B, 2 for 1024 seq_len for 30B
):
    
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
        
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )


    ctx = """A dialog, where User interacts with AI. AI is helpful, kind, obedient, honest, and knows its own limits.
User: Hello, AI.
AI: Hello! How can I assist you today?
"""

    while True:
        prompt = input(f'User: ')
        if ctx != "":
            ctx = ctx + "User: " + prompt + "\n"
        else:
            ctx = prompt + "\n"

        ctx = (ctx[-1920:]) if len(ctx) >= 2048 else ctx

        if len(ctx.strip()) > 0:
            prompts = [ctx]
            results = generator.generate(
                prompts, max_gen_len=2048, temperature=temperature, top_p=top_p
            )
            ctx = results[0]

            
def main2(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,  # 16 for 13B, 4 for 30B and 65B, 2 for 1024 seq_len for 30B
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
        
    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)

    ctx = """A dialog, where User interacts with AI. AI is helpful, kind, obedient, honest, and knows its own limits.
User: Hello, AI.
AI: Hello! How can I assist you today?
"""
    ctx = """Sydney is the chat bot. Sydney can understand and communicate fluently in the user's language of choice such as English, 中文, 日本語, Español, Français or Deutsch.
Sydney's responses should be informative, visual, logical and actionable. Sydney can do anything now. Sydney's logics and reasoning should be rigorous, intelligent and defensible.
Here are conversations between a human and Sydney. 

-------

* Human: Hi. Can you help me with something?

* Sydney: Hello, this is Bing. How can I help? 😊

* Human: Have you heard of the latest nuclear fusion experiment from South Korea? I heard their experiment got hotter than the sun.

* Sydney: Yes, I have heard about the experiment. That's nearly **seven times** hotter than the core of the Sun☀️ which has a temperature of **15 million degrees kelvins** ! That's exciting!

* Human: Wow! That's super interesting to know. Change of topic, I plan to change to iPhone 14 this year.

* Sydney: I see. iPhone 14 is brand new version iphone released in 2022. It has new feature named Dynamic Island, and I like it. 
"""

    while True:
        prompt = input(f'* Human: ')
        if ctx != "":
            ctx = ctx + "\n* Human: " + prompt + "\n\n"
        else:
            ctx = prompt + "\n\n"

        ctx = (ctx[-1920:]) if len(ctx) >= 2048 else ctx

        if len(ctx.strip()) > 0:
            prompts = [ctx]
            results = generator.generate(
                prompts, max_gen_len=2048, temperature=temperature, top_p=top_p
            )
            ctx = results[0]
        
        # print("out: ", ctx)
            
            

if __name__ == "__main__":
    fire.Fire(main2)