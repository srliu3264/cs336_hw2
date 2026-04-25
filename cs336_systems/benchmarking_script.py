from __future__ import annotations

import argparse
import statistics
import timeit

import torch

# import torch.nn as nn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

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
    p.add_argument(
        "--lr", type=float, default=5e-3
    )  # irrelevant, so i put best lr from hw1
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
) -> None:
    # x:(..., batch_size, seq_len), y:(...,batch_size, seq_len)
    if mode == "forward":
        with torch.no_grad():  # do not save grads
            _ = model(x)
        return

    logits = model(x)  # (..., batch_size, seq_len, vovab_size)
    loss = cross_entropy(
        logits.view(-1, logits.shape[-1]), y.view(-1)
    )  # (...,batch_size*seq_len)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if mode == "forward_backward":
        return
    optimizer.step()
    return


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:  # this modification is for modal
        args = parse_args()
    torch.manual_seed(args.seed)

    conf = resolve_arg(args)
    device = args.device
    addtorch = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = addtorch[args.dtype]

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

    # warm up
    for _ in range(args.warmup):
        run_step(model, optimizer, x, y, args.mode)
        sync(device)
    print("Warm-up completed.\n")
    times: list[float] = []
    for _ in range(args.steps):
        sync(device)
        t0 = timeit.default_timer()
        run_step(model, optimizer, x, y, args.mode)
        sync(device)
        times.append(timeit.default_timer() - t0)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"[result] mean = {mean*1000 : .2f} ms, std = {std*1000 : .2f} ms \n max = {max(times)*1000: .2f} ms, min = {min(times)*1000: .2f} ms"
    )


if __name__ == "__main__":
    main()
