I want to record how I did most stuff for future reference.

Also I want to summarize the pdf along the way.

## Slack Info

1. Should always use B200 even H100 is mentioned in pdf: [link](https://stanford-cs336.slack.com/archives/C0AEU1NJWSC/p1776664989925639)
2. Mostly you should use torch.no_grad (or better still, `torch.inference_mode`) for the forward-only, unless you're asked to account for the memory saved for backward.
3. Mostly you should use torch.no_grad (or better still, torch.inference_mode) for the forward-only, unless you're asked to account for the memory saved for backward. [Link](https://stanford-cs336.slack.com/archives/C0AEU1NJWSC/p1776824327797059).
4. In `nsprofiler`, cuda runtime and gpu time mean the same thing in (b) and (c).
5. Herman gives code for running nsys on modal:

```python
@app.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && apt-get install -y wget",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
    .uv_pip_install('numpy', 'torch')
    .add_local_dir('.', '/profiling'),
    gpu='H100',
    timeout=32_400,  # 9h
)
def check_caps():
    import subprocess
    subprocess.run("nvidia-smi", shell=True, check=True)
    subprocess.run("nsys --version", shell=True, check=True)
    subprocess.run(
        "nsys profile -o /tmp/profile_result --pytorch autograd-nvtx "
        "--gpu-metrics-devices all -- python /profiling/profile_train.py",
        shell=True,
    )
```
6. discussion on image build failure:

```python
def build_image(*, include_tests: bool = False) -> modal.Image:
    image = (
            modal.Image.debian_slim(python_version="3.12")
            .run_commands(
                "apt-get update && apt-get install -y wget",
                "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
                "dpkg -i cuda-keyring_1.1-1_all.deb",
                "apt-get update",
            )
            .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
            .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
            .uv_sync()
            .add_local_python_source("cs336_systems")
        )
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image
```

## 1 Assignment Overview

Six things to implement: (1) benchmarking/profiling harness, (2) activation checkpointing, (3) FlashAttention-2 Triton kernel, (4) DDP, (5) optimizer state sharding, (6) FSDP.

I am using the **staff**'s `cs336_basics`. Maybe later I should try using my own codes.

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

### benchmarking_script

Implementation: `cs336_systems/benchmarking_script.py` + Modal `cs336_systems/benchmarking_script_modal.py`.

It is not hard to write `benchmarking_script.py`: basically parse the arguement (take the template above into account), then implement as instructions. Remeber to sync though.

- For `mode=forward`, we use `torch.no_grad()` so no autograd graph build. (aligning with Slack 2)
- sync before `t0`  

Then adpat the code to modal:
- again following the modal guide to set up enviroment
- copy `modal_utils` to `cs336_systems`
- adapt the code to modal version.
- solve the image build failure issue using Slack 6, namely `.add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)`.

There are some bugs introduced by me due to poor considerations:

- has to modify `main()` to `main(args: argsparse.Namespace| None = None)` and `if args is None: ` this is for easier adaption for modal.
- in `BasicTransformerLM` I should use keyword arg.

I ran 

```bash
for size in small medium large xl 10B; do
    for mode in forward forward_backward full_step; do
      uv run modal run --detach cs336_systems/benchmarking_script_modal.py \
        --size $size --mode $mode --warmup 5 --steps 10
    done
  done
```

and then I realized it is a horrible idea:
1. it seems not to run in parallel;
2. I forgot to output results in a file but all results are in std so I manually copy-pasted results.

Anyway, the results for B200 fp32, ctx=512, bs=4, 10 timed steps:

| size   | params  | fwd (ms)        | fwd+bwd (ms)    | full step (ms)  |
| ------ | ------- | --------------- | --------------- | --------------- |
| small  | 128.6M  | 16.32 ± 0.01    | 48.59 ± 0.03    | 56.44 ± 0.04    |
| medium | 423.2M  | 47.00 ± 0.02    | 140.29 ± 0.80   | 157.86 ± 0.67   |
| large  | 969.4M  | 106.52 ± 0.03   | 313.79 ± 0.38   | 344.94 ± 1.04   |
| xl     | 3406.8M | 291.46 ± 0.02   | 866.80 ± 0.27   | 947.34 ± 0.41   |
| 10B    | 12832.8M| 945.99 ± 0.14   | 2810.50 ± 0.26  | **OOM**         |


For 10B, got OOM. But this is expected, as said in [Slack](https://stanford-cs336.slack.com/archives/C0AEU1NJWSC/p1776408673008639).

For (c), ran `uv run modal run cs336_systems/benchmarking_script_modal.py --size small --mode full_step --warmup 0 --steps 10`, and then for $w=1,2$.

| w | mean (ms) | std (ms) | min   | max    |
| - | --------- | -------- | ----- | ------ |
| 0 | 84.03     | 88.27    | 55.66 | 335.23 |
| 1 | 55.64     | 0.14     | 55.56 | 56.04  |
| 2 | 56.06     | 0.04     | 55.98 | 56.11  |
| 5 | 56.44     | 0.04     | 56.35 | 56.50  |


### nsight-systems

I am using `ArchLinux`. Need to run `sudo pacman -Sy` to update package base first, and then `sudo pacman -S nsight-systems`. I also noticed that there are AUR but I did not investigate it (a little bit sus).

```bash
nsys --version
```

Outputs: `NVIDIA Nsight Systems version 2025.6.3.541-256337736014v0`. This is same as the webpage time tag. 

Now I start to annotate the codes. Then I will get more familiar with nsys CLI.

Pdf says: you should use NVTX ranges to ignore the warm-up steps in your benchmarking script by applying an `--nvtx-capture` filter on the nvtx label in the profile 

When editing staff's code, I realize I should also use

```python
from jaxtyping import Bool, Float
from torch import Tensor
from beartype import beartype #for shape check
```

and `Q: Float[Tensor, " ... queries d_k"]` to indicate shape rather than manually comment.

Then write a modal driver. The trick is to use `subprocess.run(cmd, check=True)` and cmd is the list of strings whose join is the shell command we want to run.
Also write a new image build using Slack 6 to include nsight-system.

Add
```toml
"beartype>=0.18",                                                                                                                                                │
"jaxtyping>=0.2",                                                                                                                                                │
"einops>=0.7"
```
to `pyproject.toml`.

Then run `uv run modal run --detach cs336_systems/nsys_profile_modal.py --size small --context-length 256 --mode full_step` for different sizes and context lengths.

Then it is standard modal to download results:

```bash
mkdir -p nsys_reports
uv run modal volume get systems-srliu nsys/ ./nsys_reports/
nsys-ui ./nsys_reports/nsys/nsys_small_ctx256_full_step.nsys-rep
```

For each of the 6 reports: pick a post-warmup step (e.g. `step_5`), filter the Stats System View → CUDA GPU Kernel Summary by the relevant NVTX range, sort by Total Time descending. Use the GUI ("Filter and Zoom in" by right-clicking an NVTX block) or CLI:

I am looking at step 5 after warm up. The forward time only takes 34~ms for small size regardless of ctx and  filter using NVTX in threads and NVTX in CUDA HW are dramatically different (the latter is about 260ms for `ctx=4096`).  Namely, I get the following


| size  | ctx  | step total (ms) | forward (ms) | backward (ms) | optimizer (ms) | match §2.1.3 fwd? |
| ----- | ---- | --------------- | ------------ | ------------- | -------------- | ----------------- |
| small | 256  |     118.325            |    38.411          |      45.996         |      31.974          |                   |
| small | 1024 |       121.798          |     33.708         |     47.668          |      38.812          |                   |
| small | 4096 |       783.986          |     34.638         |        393.401       |         354.265       |                   |
| large | 256  |         368.017        |     101.909         |       151.479        |       109.927         |                   |
| large | 1024 |       769.165          |      172.608        |      382.000         |      211.321          |                   |
| large | 2048 |        OOM          |      OOM        |        OOM       |        OOM        |       |
| large | 4096 |       OOM          |      OOM        |        OOM       |        OOM        |       |

Also filter using NVTX "backward" in threads will overlap a lot  with NVTX "forward" in CUDA HW.

Then I noticed [Slack question](https://stanford-cs336.slack.com/archives/C0AEU1NJWSC/p1776475093163179), which inspires me to add more sync to separate foward, backward, optimizer in `run_step`.


| size  | ctx  | step total (ms) | forward (ms) | backward (ms) | optimizer (ms) | match §2.1.3 fwd? |
| ----- | ---- | --------------- | ------------ | ------------- | -------------- | ----------------- |
| small | 256  |     118.325            |    38.411          |      45.996         |      31.974          |                   |
| small | 1024 |       121.798          |     33.708         |     47.668          |      38.812          |                   |
| small | 4096 |       783.986          |     34.638         |        393.401       |         354.265       |                   |
| large | 256  |         368.017        |     101.909         |       151.479        |       109.927         |                   |
| large | 1024 |       769.165          |      172.608        |      382.000         |      211.321          |                   |
| large | 2048 |        OOM          |      OOM        |        OOM       |        OOM        |       |
| large | 4096 |       OOM          |      OOM        |        OOM       |        OOM        |       |



| size  | ctx  | top fwd kernel (name) | invocations | total ms | % of fwd kernel time |
| ----- | ---- | --------------------- | ----------- | -------- | -------------------- |
| small | 256  |                       |             |          |                      |
| small | 1024 |                       |             |          |                      |
| small | 4096 |                       |             |          |                      |
| large | 256  |                       |             |          |                      |
| large | 1024 |                       |             |          |                      |
| large | 4096 |                       |             |          |                      |


| size  | ctx  | top fwd+bwd kernel | same as fwd? |
| ----- | ---- | ------------------ | ------------ |
| small | 256  |                    |              |
| small | 1024 |                    |              |
| small | 4096 |                    |              |
| large | 256  |                    |              |
| large | 1024 |                    |              |
| large | 4096 |                    |              |


Non-matmul kernels that consume non-trivial CUDA time 

```
small/256:  -
small/1024: -
small/4096: -
large/256:  -
large/1024: -
large/4096: -
```


| size  | ctx  | matmul% in fwd | matmul% in full step | which non-matmul kernels grow? |
| ----- | ---- | -------------- | -------------------- | ------------------------------ |
| small | 256  |                |                      |                                |
| small | 1024 |                |                      |                                |
| small | 4096 |                |                      |                                |
| large | 256  |                |                      |                                |
| large | 1024 |                |                      |                                |
| large | 4096 |                |                      |                                |



Filter by NVTX = `computing attention scores` / `computing softmax` / `final matmul`.


| size  | ctx  | scores matmul (ms) | softmax (ms) | final matmul (ms) |
| ----- | ---- | ------------------ | ------------ | ----------------- |
| small | 256  |                    |              |                   |
| small | 1024 |                    |              |                   |
| small | 4096 |                    |              |                   |
| large | 256  |                    |              |                   |
| large | 1024 |                    |              |                   |
| large | 4096 |                    |              |                   |


FLOPs reference (per forward, per layer, both matmuls vs softmax):
- scores matmul: $2 \cdot B \cdot H \cdot T^2 \cdot d_k$ FLOPs
- final matmul: $2 \cdot B \cdot H \cdot T^2 \cdot d_k$ FLOPs
- softmax: $\sim 5 \cdot B \cdot H \cdot T^2$ 

