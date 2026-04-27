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


| size  | ctx  | step total (ms) | forward (ms) | backward (ms) | optimizer (ms) | match fwd? |
| ----- | ---- | --------------- | ------------ | ------------- | -------------- | ----------------- |
| small | 256  |     118.325            |    38.411        |      45.996         |      31.974          |      |
| small | 1024 |       121.798          |     33.708         |     47.668          |      38.812          |                   |
| small | 4096 |       783.986          |     34.638         |        393.401       |         354.265       |                   |
| small | 8192 |       OOM          |      OOM        |        OOM       |        OOM        |       |
| large | 256  |         368.017        |     101.909         |       151.479        |       109.927         |   |
| large | 1024 |       769.165          |      172.608        |      382.000         |      211.321          |           |
| large | 2048 |        OOM          |      OOM        |        OOM       |        OOM        |       |
| large | 4096 |       OOM          |      OOM        |        OOM       |        OOM        |       |

Also filter using NVTX "backward" in threads will overlap a lot  with NVTX "forward" in CUDA HW.

Then I noticed [Slack question](https://stanford-cs336.slack.com/archives/C0AEU1NJWSC/p1776475093163179), which inspires me to add more sync to separate foward, backward, optimizer in `run_step`.

Now with sync isolation, I reran 6 experiments and their ablation. Now two filters NVTX basically match. (number from step5 i=5)

| size  | ctx  | step total (ms) | forward (ms) | backward (ms) | optimizer (ms) | fwd (standard lib) |
| ----- | ---- | --------------- | ------------ | ------------- | -------------- | ----------------- |
| small | 256  |           108.617      |    33.419          |      45.134         |      28.523          |  9.67±0.04                 |
| small | 1024 |       174.219          |     55.890         |     74.311          |      42.775         |  37.07±0.04                 |
| small | 4096 |       821.933         |     265.225         |        510.713       |         43.736       |    258.01±0.03               |
| large | 256  |         511.931        |     164.922         |       215.738        |       126.523         |    62.23±0.04               |
| large | 1024 |       755.812          |      232.132        |      436.022         |      82.041          |   229.35±0.12                |
| large | 2048 |        OOM          |      OOM        |        OOM       |        OOM        |       |
| large | 4096 |       OOM          |      OOM        |        OOM       |        OOM        |       |



| size  | ctx  | top fwd kernel (name) | instances | total ms | % of fwd kernel time |
| ----- | ---- | --------------------- | ----------- | -------- | -------------------- |
| small | 256  |    `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`                   |   24          |    2.746      |     31.8%                 |
| small | 1024 |    `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`                   |   60          |    10.234      |          28.5%            | 
| small | 4096 |  `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_tnn_align1_bias_f32_relu`                     |     37        |   70.347       |         26.9%             |
| large | 256  |   `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_tnn_align1_bias_f32_relu`                    |  180           |    26.909      |              45.0%        |
| large | 1024 |  `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_tnn_align1_bias_f32_relu`                     |    109         |   84.128       |         36.9%             |




| size  | ctx  | top fwd+bwd kernel                                                                                                                  | same as fwd?                                 |
| ----- | ---- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| small | 256  | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_nnn_align1_bias_f32_relu`                                          | **No** — fwd top was `64x64x16 tnn_align1`   |
| small | 1024 | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu`                                           | **Different layout** — fwd was `tnn_align1`  |
| small | 4096 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | **No** — non-matmul (softmax-grad division) takes the lead |
| large | 256  | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_nnn_align1_bias_f32_relu`                                           | **Different layout** — fwd was `tnn_align1`  |
| large | 1024 | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_ntn_align1_bias_f32_relu`                                           | **No** — fwd top was `128x128x16 tnn_align1` |


- **DivFunctor**: `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` — softmax denominator divide ($P/Z$) and the $1/\sqrt{d_k}$ scaling.
- **where_kernel_impl**: `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::<unnamed>::where_kernel_impl(at::TensorIterator &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 11)]::operator ()() const::[lambda(bool, float, float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` — causal-mask `masked_fill` (`torch.where(mask, scores, -inf)`).
- **CUDAFunctor_add**: `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` — residual adds.

Plus `exp_kernel_cuda` (softmax numerator), `reduce_kernel<MaxOps>` (softmax max-shift), `reduce_kernel<MeanOps>` / `reduce_kernel<func_wrapp_t>` (RMSNorm $E[x^2]$ and softmax sum-reduce), and `CatArrayBatchedCopy` (concat MHSA heads).


| size  | ctx  | matmul% | DivFunctor | where_kernel | CUDAFunctor_add | reduce(MaxOps) | exp_kernel | notes                          |
| ----- | ---- | ------- | ---------- | ------------ | --------------- | -------------- | ---------- | ------------------------------ |
| small | 256  | 78.3    | (top elementwise was MulFunctor 4.9%) |     | 1.8%            | 1.3%           |           | cat 1.8%, reduce(MeanOps) 1.3% |
| small | 1024 | 69.7    | 5.2%       | 4.6%         | 4.5%            | 2.1%           | 2.0%       | reduce(sum) 1.7%               |
| small | 4096 | 49.4    | 11.5%      | 10.7%        | 9.9%            | 3.3%           | 4.3%       | reduce(sum) 2.4%               |
| large | 256  | 87.3    | 0.9%       | —            | 0.9%            | 0.8%           | —          | top elt: MulFunctor 2.8%, cat 1.1% |
| large | 1024 | 76.8    | 4.1%       | 3.6%         | 3.5%            | 1.6%           | 1.5%       |                                |



| size  | ctx  | matmul% in fwd | matmul% in full step | what increases in full step                                |
| ----- | ---- | -------------- | -------------------- | ------------------------------------------------------ |
| small | 256  | 78.3%          | 64.6%                | AdamW elementwise (`_foreach_*`), masked_fill backward |
| small | 1024 | 69.7%          | 64.7%                | AdamW elementwise + softmax-bwd                        |
| small | 4096 | 49.4%          | 47.0%                | softmax-bwd elementwise dominate                       |
| large | 256  | 87.3%          | 74.6%                | AdamW elementwise                                      |
| large | 1024 | 76.8%          | 72.2%                | AdamW elementwise + softmax-bwd                        |



| size  | ctx  | scores matmul | softmax | final matmul | softmax/scores | softmax/final |
| ----- | ---- | ------------- | ------- | ------------ | -------------- | ------------- |
| small | 256  | 0.005         | 0.003   | 0.003        | 0.6            | 1.0           |
| small | 1024 | 0.030         | 0.030   | 0.020        | ~1.0           | ~1.5          |
| small | 4096 | 0.330         | 0.520   | 0.300        | 1.6            | 1.7           |
| large | 256  | 0.003         | 0.005   | 0.005        | ~1.5           | ~1.0          |
| large | 1024 | 0.040         | 0.050   | 0.030        | 1.3            | 1.7           |


I realized it is false above to run nsys with full step and then compare forward.

I directly added these to LaTeX and I am lazy to paste here.

### mixed precision

I copied the code to ipynb and ran it.

Look at `torch.autocast`.

Then as in pdf, wrap forward with autocast context (use `nullcontext` to turn off) and add new argument to parse.

I also finally add a new arguement to output results in a csv file. The only hard part is to carefully write os to mkdir and also for a new file, need to write column names. For modal driver, need to commit to user volume defined in utils.

```bash
uv run modal run --detach cs336_systems/benchmarking_script_modal.py --size small --mode forward_backward --results-file /root/data/mp_small.csv
```

### Memory

Samilar to mixed precision:

1. add wrappers in `benchmarking_script.py`
2. add arguments (in remote, args namespace, main) and commit to user volumn

Experiment:
```bash
uv run modal run --detach cs336_systems/benchmarking_script_modal.py --size xl --mode forward --steps 1 --context-length 2048 --memory-profile --memory-snapshot-path /root/data/mem_xl_fwd_ctx2048.pickle
```

for xl, and for forward, full step, steps=1, ctx = 256, 2048.

ctx=2048 all OOM except for forward with mp bf16. This is also reported in [Slack](https://stanford-cs336.slack.com/archives/C0AEU1NJWSC/p1776747260518119).

Then I went to [https://docs.pytorch.org/memory_viz](memory viz). Take screenshots. Change Detail to be 202/1999 (I can not exactly dragged it to 200 or 199).

Then it outputs

```txt
141 Addr: b'14ae60000000_1, Size: 2.0GiB (2147483648 bytes) allocation, Total memory used after allocation: 21.2GiB (22735054848 bytes), Compile context: N/A, stream 0, pool_id (0, 0), timestamp Sun Apr 26 2026 00:03:44 GMT-0700 (Pacific Daylight Time)
CUDACachingAllocator.cpp:0:c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc(unsigned long, CUstream_st*)
:0:c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::malloc(void**, signed char, unsigned long, CUstream_st*)
:0:c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::allocate(unsigned long)
??:0:at::detail::empty_generic(c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType, std::optional<c10::MemoryFormat>)
??:0:at::detail::empty_cuda(c10::ArrayRef<long>, c10::ScalarType, std::optional<c10::Device>, std::optional<c10::MemoryFormat>)
??:0:at::detail::empty_cuda(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, std::optional<c10::MemoryFormat>)
??:0:at::detail::empty_cuda(c10::ArrayRef<long>, c10::TensorOptions const&)
RegisterCUDA_0.cpp:0:at::(anonymous namespace)::create_out(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions const&)
RegisterCUDA_0.cpp:0:at::(anonymous namespace)::structured_div_out_functional::set_output_raw_strided(long, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions, c10::ArrayRef<at::Dimname>)
??:0:at::TensorIteratorBase::allocate_or_resize_outputs()
??:0:at::TensorIteratorBase::build(at::TensorIteratorConfig&)
??:0:at::TensorIteratorBase::build_borrowing_binary_float_op(at::TensorBase const&, at::TensorBase const&, at::TensorBase const&)
RegisterCUDA_0.cpp:0:at::(anonymous namespace)::wrapper_CUDA_div_Tensor(at::Tensor const&, at::Tensor const&)
RegisterCUDA_0.cpp:0:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (at::Tensor const&, at::Tensor const&), &at::(anonymous namespace)::wrapper_CUDA_div_Tensor>, at::Tensor, c10::guts::typelist::typelist<at::Tensor const&, at::Tensor const&> >, at::Tensor (at::Tensor const&, at::Tensor const&)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&)
??:0:at::_ops::div_Tensor::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&)
VariableType_1.cpp:0:torch::autograd::VariableType::(anonymous namespace)::div_Tensor(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&)
VariableType_1.cpp:0:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::DispatchKeySet, at::Tensor const&, at::Tensor const&), &torch::autograd::VariableType::(anonymous namespace)::div_Tensor>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, at::Tensor const&> >, at::Tensor (c10::DispatchKeySet, at::Tensor const&, at::Tensor const&)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&)
??:0:at::_ops::div_Tensor::call(at::Tensor const&, at::Tensor const&)
python_variable_methods.cpp:0:torch::autograd::THPVariable_div(_object*, _object*, _object*)
python_variable_methods.cpp:0:_object* torch::autograd::TypeError_to_NotImplemented_<&torch::autograd::THPVariable_div>(_object*, _object*, _object*)
??:0:PyType_GetModule
??:0:Py_GetRecursionLimit
??:0:PyNumber_Add
??:0:PyNumber_TrueDivide
??:0:_PyMem_SetupAllocators
<repeats 9 times>
??:0:PySequence_DelItem
??:0:_PyMem_SetupAllocators
??:0:_PyInterpreterState_SetRunningMain
??:0:Py_RunMain
??:0:Py_BytesMain
??:0:__libc_init_first
/.uv/cs336-basics/cs336_basics/nn_utils.py:7:softmax
/.uv/cs336-basics/cs336_basics/model.py:432:scaled_dot_product_attention
/.uv/cs336-basics/cs336_basics/model.py:520:forward
/.uv/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1790:_call_impl
/.uv/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1779:_wrapped_call_impl
/.uv/cs336-basics/cs336_basics/model.py:382:forward
/.uv/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1790:_call_impl
/.uv/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1779:_wrapped_call_impl
/.uv/cs336-basics/cs336_basics/model.py:253:forward
/.uv/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1790:_call_impl
/.uv/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1779:_wrapped_call_impl
/root/cs336_systems/benchmarking_script.py:162:run_step
/root/cs336_systems/benchmarking_script.py:261:main
/root/benchmarking_script_modal.py:68:benchmark_remote
/pkg/modal/_runtime/container_io_manager.py:225:call_function_sync
/pkg/modal/_container_entrypoint.py:172:run_input_sync
/pkg/modal/_container_entrypoint.py:247:call_function
/pkg/modal/_container_entrypoint.py:388:main
/pkg/modal/_container_entrypoint.py:406:<module>
<frozen runpy>:88:_run_code
<frozen runpy>:198:_run_module_as_main
```


Apprarently softmax is the key countribution.

For ctx 128, no item contributes 10%.

### nsys memory

Add new argument to toggle if we should annotate transformer block (like attention we did before).

Use `--cuda-memory-usage=true` for nsys.

`uv run modal run --detach cs336_systems/nsys_profile_modal.py --size xl --context-length 1024 --mode forward_backward --warmup 5 --steps 1 --annotate-blocks --run-name mem_nsys_xl_ctx1024`

I am deeply confused on how to annotate backward phase. I only annotated backward and I don't know what to read on GUI to answer the question. Then Gemini pointed to me that i need to set correct env:  `env["PYTORCH_CUDA_ALLOC_CONF"] ="backend:cudaMallocAsync"`.

I realize Gemini gives useless info.

The mistake I made is: I should still use NVTX in threads to filter, and then hover over the edge to see the numbers (since the change Delta is really small 2GB compared to 100GB)!

Wasted 2hrs on this reading GPT, Gemini, and Torch... Should wait for OH!

Then I found it is impossible to find top5 looking at the stats view. I learnt from Slack that I should use `nsys stats --report cuda_api_sum` and then use sql to query `CUDA_GPU_MEMORY_USAGE_EVENTS` table. I will do this later.

(f) turns out to be very fun! I am happy to see the theoretical value matches the emperical value extracted from nsys GUI.


## Single GPU

Learnt checkpoint trick. This is just a math problem lol.

Add a new argument checkpoint-block-szie, and then wrap blocks in checkpoints. similar to before, edit the functions; then add args / modify arg namespaces for main and remote in modal version.

A subtle point I learnt is that we had better have `reset_peak_memory_stats()`.

`uv run modal run --detach cs336_systems/benchmarking_script_modal.py --size xl --context-length 2048 --mode forward_backward --steps 5 --checkpoint-block-size 1`

## Flash Attention!

###  attention benchmarking

1. Forward timing under `no_grad`
2. I still use timeit since I don't need a profiler to look at each component for this problem.
3. This is important (learnt from Claude): `gc.collect()` `torch.cuda.empty_cache()` to free memory so that one OOM will not stop further experiment. But Claud's suggestion on using `del locals()["name of var"]` is sus. I would simply use del.
4. be careful and set device to be cuda everywhere. 
5. csv.DictWriter is very useful. I should have done this in section2. This is much easier.

`uv run modal run --detach cs336_systems/attention_benchmark_modal.py --results-file /root/data/attn_uncompiled.csv`

emmm should not use either subprocess or sys.argv. simply use optional args in main and patch.

I also forgot to pass an absolute path.
