from __future__ import annotations

import argparse
import math
import statistics
import timeit
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx
from beartype import beartype  # for shape check

# import torch.nn as nn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW
from einops import einsum

# for type hint
from jaxtyping import Bool, Float
from torch import Tensor


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        out = einsum(
            attention_weights, V, "... query key, ... key d_v ->  ... query d_v"
        )
    return out


MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}
MODES = ("forward", "forward_backward", "full_step")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=str, default=None, choices=list(MODEL_SIZES))
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--rope_theta", type=float, default=10_000.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--mode", type=str, default="full_step", choices=MODES)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--annotate_attention", action="store_true", default=False)
    p.add_argument("--mixed_precision", action="store_true", default=False)
    p.add_argument(
        "--mixed_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    p.add_argument(
        "--lr", type=float, default=5e-3
    )  # irrelevant, so i put best lr from hw1
    p.add_argument(  # append to a csv file
        "--results_file",
        type=str,
        default=None,
    )
    p.add_argument("--memory_profile", action="store_true", default=False)
    p.add_argument("--memory_snapshot_path", type=str, default=None)
    return p.parse_args()


def resolve_arg(args: argparse.Namespace) -> dict:
    """Fill model configurations using predefined MODEL_SIZES and check if necessary arguments are complete."""
    conf = {}
    if args.size is not None:
        conf = dict(MODEL_SIZES[args.size])
    params = ["d_model", "d_ff", "num_layers", "num_heads"]
    for par in params:
        value = getattr(args, par)
        if value is not None:
            conf[par] = value
    for par in params:
        value = conf.get(par, None)
        if value is None:
            raise ValueError(f"missing {par}")
    return conf


def sync(device: str) -> None:
    """
    CUDA calls are asynchronous, so we need to wait for all scheduled GPU kernels to complete, allowing us to get more accurate measurements of CUDA kernel runtime
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return


def run_step(
    model,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    device="cuda",
    amp_ctx=None,
) -> None:
    if amp_ctx is None:
        amp_ctx = nullcontext()
    # x:(..., batch_size, seq_len), y:(...,batch_size, seq_len)
    if mode == "forward":
        with nvtx.range("forward"):
            with torch.no_grad():  # do not save grads
                with amp_ctx:
                    _ = model(x)
            sync(device)
            return
    with nvtx.range("forward"):
        with amp_ctx:
            logits = model(x)  # (..., batch_size, seq_len, vovab_size)
        sync(device)
    loss = cross_entropy(
        logits.view(-1, logits.shape[-1]), y.view(-1)
    )  # (...,batch_size*seq_len)
    optimizer.zero_grad(set_to_none=True)
    with nvtx.range("backward"):
        loss.backward()
        sync(device)
    if mode == "forward_backward":
        return
    with nvtx.range("step"):
        optimizer.step()
        sync(device)
    return


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:  # this modification is for modal
        args = parse_args()

    torch.manual_seed(args.seed)

    if args.annotate_attention:
        import cs336_basics.model

        cs336_basics.model.scaled_dot_product_attention = (
            annotated_scaled_dot_product_attention
        )
    conf = resolve_arg(args)
    device = args.device
    addtorch = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = addtorch[args.dtype]

    # add autocast context
    if args.mixed_precision:
        amp_dtype = addtorch[args.mixed_dtype]
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()

    # initialize model and optimizer
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        **conf,
    ).to(device=device, dtype=dtype)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # generate a batch of data
    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )
    y = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Hyperparameters] model_arch = {conf}, ctx = {args.context_length}, bs = {args.batch_size}, dtype = {args.dtype}, device = {device}"
    )

    print(
        f"params={n_params/1e6:.1f}M, mode={args.mode}, warmup={args.warmup}, steps={args.steps}"
    )

    if args.mixed_precision:
        print(f"mixed precision: {args.mixed_dtype}")

    # warm up
    with nvtx.range("warmup"):
        for _ in range(args.warmup):
            run_step(model, optimizer, x, y, args.mode, device, amp_ctx=amp_ctx)
            sync(device)
    print("Warm-up completed.\n")

    if args.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    # run steps
    times: list[float] = []
    for i in range(args.steps):
        sync(device)
        with nvtx.range(f"step {i}"):
            t0 = timeit.default_timer()
            run_step(model, optimizer, x, y, args.mode, device, amp_ctx=amp_ctx)
            sync(device)
            times.append(timeit.default_timer() - t0)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"[result] mean = {mean*1000 : .2f} ms, std = {std*1000 : .2f} ms \n max = {max(times)*1000: .2f} ms, min = {min(times)*1000: .2f} ms"
    )
    if args.memory_profile:
        if args.memory_snapshot_path is None:
            raise ValueError(
                "--memory_snap_path did not receive paths for outputpickle file"
            )
        import os

        os.makedirs(
            os.path.dirname(os.path.abspath(args.memory_snapshot_path)), exist_ok=True
        )
        torch.cuda.memory._dump_snapshot(args.memory_snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"[memory] snapshot pickle saved to {args.memory_snapshot_path}")
    if args.results_file is not None:
        import os

        os.makedirs(
            os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True
        )  # get ./dir name from file name and then mkdir -p
        new_file = not os.path.exists(args.results_file)
        with open(args.results_file, "a") as f:
            if new_file:
                f.write(
                    "size,mode,mixed_precision,mixed_dtype,context_length,"
                    "batch_size,warmup,steps,n_params,"
                    "mean_ms,std_ms,max_ms,min_ms\n"
                )
            f.write(
                f"{args.size},{args.mode},{args.mixed_precision},"
                f"{args.mixed_dtype if args.mixed_precision else 'N/A'},"
                f"{args.context_length},{args.batch_size},"
                f"{args.warmup},{args.steps},{n_params},"
                f"{mean*1000:.4f},{std*1000:.4f},"
                f"{max(times)*1000:.4f},{min(times)*1000:.4f}\n"
            )
        print(f"[result] appended to {args.results_file}")


if __name__ == "__main__":
    main()
