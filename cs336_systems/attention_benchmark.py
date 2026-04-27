"to solve problem 4.1 attention benchmarking"

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import timeit
from itertools import product
from os.path import exists

import torch
from cs336_basics.model import scaled_dot_product_attention

D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
BATCH_SIZE = 8
WARMUP = 5
ITERS = 100


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--results_file", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def gen_qkv(
    B: int, T: int, d: int, device: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Q = torch.randn(B, T, d, device=device, requires_grad=True)
    K = torch.randn(B, T, d, device=device, requires_grad=True)
    V = torch.randn(B, T, d, device=device, requires_grad=True)
    return Q, K, V


def time_forward(attn_fn, Q, K, V, iters: int):
    times = []
    with torch.no_grad():  # remember tip from Slack
        for _ in range(iters):
            sync()
            t0 = timeit.default_timer()
            _ = attn_fn(Q, K, V)
            sync()
            times.append(timeit.default_timer() - t0)
    return statistics.mean(times) * 1000.0  # unit:s


def time_backward(attn_fn, Q, K, V, iters):
    times = []
    mem_before_bwd = 0.0
    for i in range(iters):
        # before backward: zero grad, sync
        if Q.grad is not None:
            Q.grad = None
        if K.grad is not None:
            K.grad = None
        if V.grad is not None:
            V.grad = None

        out = attn_fn(Q, K, V)
        loss = out.sum()
        sync()
        if i == 0:
            mem_before_bwd = torch.cuda.memory_allocated() / 1024**2
        t0 = timeit.default_timer()
        loss.backward()
        sync()
        times.append(timeit.default_timer() - t0)
    return statistics.mean(times) * 1000.0, mem_before_bwd  # unit: s, MiB


def find_oom(d, T, attn_fn, device: str | None = None):
    try:
        Q, K, V = gen_qkv(BATCH_SIZE, T, d, device)
        # warm up
        for _ in range(WARMUP):
            out = attn_fn(Q, K, V)
            out.sum().backward()
            Q.grad = None
            K.grad = None
            V.grad = None
        sync()
        torch.cuda.reset_peak_memory_stats()
        fwd_ms = time_forward(attn_fn, Q, K, V, ITERS)
        bwd_ms, mem_before_bwd = time_backward(attn_fn, Q, K, V, ITERS)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return dict(
            d=d,
            T=T,
            fwd_ms=fwd_ms,
            mem_before_bwd_mib=mem_before_bwd,
            bwd_ms=bwd_ms,
            peak_mib=peak,
            status="success",
        )
    except torch.cuda.OutOfMemoryError:
        return dict(
            d=d,
            T=T,
            fwd_ms=None,
            mem_before_bwd_mib=None,
            bwd_ms=None,
            peak_mib=None,
            status="OOM",
        )
    finally:  # frees memory
        try:
            del Q, K, V, loss, out
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def main(args: argparse.Namespace | None = None):
    if args is None:
        args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("Need CUDA!")
    if not args.results_file:
        raise ValueError("Give a path to result file to save CSV!")

    device = "cuda"
    torch.manual_seed(args.seed)

    attn_fn = scaled_dot_product_attention

    if args.compile:
        attn_fn = torch.compile(scaled_dot_product_attention)
        print("[torch.compile] JIT compiled attnfn")

    rows: list[dict] = []
    for d, T in product(D_MODELS, SEQ_LENS):
        r = find_oom(d, T, attn_fn, device)
        rows.append(r)

    if args.results_file:
        import os

        os.makedirs(
            os.path.dirname(os.path.abspath(args.results_file)) or ".", exist_ok=True
        )
        with open(args.results_file, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "d",
                    "T",
                    "fwd_ms",
                    "mem_before_bwd_mib",
                    "bwd_ms",
                    "peak_mib",
                    "status",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[saved results] in {args.results_file}")


if __name__ == "__main__":
    main()
