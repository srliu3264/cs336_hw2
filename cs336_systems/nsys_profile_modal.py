import os
import subprocess
from pathlib import Path

from cs336_systems.modal_utils import (
    DATA_PATH,
    VOLUME_MOUNTS,
    app,
    build_image_nsys,
    user_volume,
)

image = build_image_nsys()
NSYS_DIR = DATA_PATH / "nsys"


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    cpu=8,
    memory=65536,
    timeout=86400,  # 24hr
)
def nsys_remote(
    run_name: str,
    size: str = "small",
    context_length: int = 512,
    mode: str = "full_step",
    warmup: int = 5,
    steps: int = 10,
    batch_size: int = 4,
    dtype: str = "float32",
    annotate_attention: bool = True,
    annotate_blocks: bool = False,
):
    out_dir = Path("/root") / NSYS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / run_name  # nsys appends .nsys-rep automatically

    cmd = [  # from pdf
        "nsys",
        "profile",
        "-o",
        str(out_path),
        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        "--pytorch=functions-trace,autograd-shapes-nvtx",
        "--cudabacktrace=all",
        "--python-backtrace=cuda",
        "--gpu-metrics-devices=0",
        "--cuda-memory-usage=true",  # for (f): records cudaMalloc/cudaFree in timeline
        "--force-overwrite=true",
        "--",
        "python",
        "-m",
        "cs336_systems.benchmarking_script",
        "--size",
        size,
        "--context_length",
        str(context_length),
        "--mode",
        mode,
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
        "--batch_size",
        str(batch_size),
        "--dtype",
        dtype,
        "--device",
        "cuda",
    ]
    if annotate_attention:
        cmd.append("--annotate_attention")
    if annotate_blocks:
        cmd.append("--annotate_blocks")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

    subprocess.run(["which", "nsys"], check=False)
    subprocess.run(["nsys", "--version"], check=False)
    subprocess.run(["which", "python"], check=False)

    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    print("[nsys stdout]\n", result.stdout, flush=True)
    print("[nsys stderr]\n", result.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"nsys failed with exit code {result.returncode}")

    user_volume.commit()  # do not forget this...
    print(f"[nsys_remote] wrote {out_path}.nsys-rep", flush=True)


@app.local_entrypoint()
def main(
    size: str = "small",
    context_length: int = 512,
    mode: str = "full_step",
    warmup: int = 5,
    steps: int = 10,
    batch_size: int = 4,
    dtype: str = "float32",
    annotate_attention: bool = True,
    annotate_blocks: bool = False,
    run_name: str = "",
):
    """Run a single nsys profile on Modal B200.

    Usage:
        uv run modal run cs336_systems/nsys_profile_modal.py --size small --context-length 256 --mode full_step

    Defaults: warmup=5, steps=10, dtype=float32, batch_size=4, annotate_attention=True.
    Output saved to volume `systems-srliu` at data/nsys/<run_name>.nsys-rep.
    """
    if not run_name:  # define a default run name
        run_name = f"nsys_{size}_ctx{context_length}_{mode}"
    nsys_remote.remote(
        run_name=run_name,
        size=size,
        context_length=context_length,
        mode=mode,
        warmup=warmup,
        steps=steps,
        batch_size=batch_size,
        dtype=dtype,
        annotate_attention=annotate_attention,
        annotate_blocks=annotate_blocks,
    )
