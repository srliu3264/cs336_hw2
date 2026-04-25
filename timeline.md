I want to record how I did most stuff for future reference.

Also I want to summarize the pdf along the way.

# Slack Info

1. Should always use B200 even H100 is mentioned in pdf: [link](https://stanford-cs336.slack.com/archives/C0AEU1NJWSC/p1776664989925639)
2. Mostly you should use torch.no_grad (or better still, `torch.inference_mode`) for the forward-only, unless you're asked to account for the memory saved for backward.
3. Mostly you should use torch.no_grad (or better still, torch.inference_mode) for the forward-only, unless you're asked to account for the memory saved for backward. [Link](https://stanford-cs336.slack.com/archives/C0AEU1NJWSC/p1776824327797059).
4. In `nsprofiler`, cuda runtime and gpu time mean the same thing in (b) and (c).

## 1 Assignment Overview 

Six things to implement: (1) benchmarking/profiling harness, (2) activation checkpointing, (3) FlashAttention-2 Triton kernel, (4) DDP, (5) optimizer state sharding, (6) FSDP.

I am using the **staff**'s `cs336_basics`

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

- For `mode=forward`, we use `torch.no_grad()` so no autograd graph build. (aligning with Slack 2)
- sync before `t0`  

Then adpat the code to modal:
- again following the modal guide to set up enviroment
- copy `modal_utils` to `cs336_systems`
- adapt the code to modal version.
