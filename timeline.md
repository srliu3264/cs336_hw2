I want to record how I did most stuff for future reference.

Also I want to summarize the pdf along the way.

## 1 Assignment Overview 

Six things to implement: (1) benchmarking/profiling harness, (2) activation checkpointing, (3) FlashAttention-2 Triton kernel, (4) DDP, (5) optimizer state sharding, (6) FSDP.
Submission: `writeup.pdf` + `code.zip` via `test_and_make_submission.sh`.
Decision: use the **staff** `cs336-basics/` (already wired in `pyproject.toml`).

## 2 Profiling and Benchmarking

### Model Size

Common settings for all (non-leaderboard) experiments: `vocab_size=10000`, `batch_size=4`, `context_length=512` unless overridden.

| size   | d_model | d_ff  | num_layers | num_heads |
| ------ | ------- | ----- | ---------- | --------- |
| small  | 768     | 3072  | 12         | 12        |
| medium | 1024    | 4096  | 24         | 16        |
| large  | 1280    | 5120  | 36         | 20        |
| xl     | 2560    | 10240 | 32         | 32        |
| 10B    | 4608    | 12288 | 50         | 36        |

Tip: use `pandas.DataFrame.to_latex()` to auto-generate result tables.

### `benchmarking_script`

This is not hard: basically parse the arguement (take the template above into account). Remeber to sync.

- For `mode=forward`, we use `torch.no_grad()` so no autograd graph build.
- sync before `t0`  
