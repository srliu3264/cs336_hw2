import argparse

import modal

from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image, user_volume

image = build_image()


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    cpu=8,
    memory=65536,
    timeout=86400,  # 24hr
)
def benchmark_remote(
    size: str | None = "small",
    d_model: int | None = None,
    d_ff: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    vocab_size: int = 10_000,
    context_length: int = 512,
    batch_size: int = 4,
    rope_theta: float = 10_000.0,
    warmup: int = 5,
    steps: int = 10,
    mode: str = "full_step",
    dtype: str = "float32",
    seed: int = 0,
    lr: float = 5e-3,
    annotate_attention: bool = False,
    mixed_precision: bool = False,
    mixed_dtype: str = "bfloat16",
    results_file: str | None = None,
    memory_profile: bool = False,
    memory_snapshot_path: str | None = None,
    annotate_blocks: bool = False,
    checkpoint_block_size: int = 0,
    compile: bool = False,
):
    from cs336_systems.benchmarking_script import main as benchmark_main

    args = argparse.Namespace(
        size=size,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        context_length=context_length,
        batch_size=batch_size,
        rope_theta=rope_theta,
        warmup=warmup,
        steps=steps,
        mode=mode,
        device="cuda",
        dtype=dtype,
        seed=seed,
        lr=lr,
        annotate_attention=annotate_attention,
        mixed_precision=mixed_precision,
        mixed_dtype=mixed_dtype,
        results_file=results_file,
        memory_profile=memory_profile,
        memory_snapshot_path=memory_snapshot_path,
        annotate_blocks=annotate_blocks,
        checkpoint_block_size=checkpoint_block_size,
        compile=compile,
    )
    try:
        return benchmark_main(args)
    finally:
        if memory_profile:
            user_volume.commit()
        if results_file is not None:
            user_volume.commit()  # append results to csv file


@app.local_entrypoint()
def main(
    size: str = "small",
    mode: str = "full_step",
    warmup: int = 5,
    steps: int = 10,
    context_length: int = 512,
    batch_size: int = 4,
    dtype: str = "float32",
    annotate_attention: bool = False,
    mixed_precision: bool = False,
    mixed_dtype: str = "bfloat16",
    results_file: str | None = None,
    memory_profile: bool = False,
    memory_snapshot_path: str | None = None,
    annotate_blocks: bool = False,
    checkpoint_block_size: int = 0,
    compile: bool = False,
):
    """
    uv run modal run cs336_systems/benchmarking_script_modal.py --size small --mode full_step --warmup 5 --steps 10
    """
    result = benchmark_remote.remote(
        size=size,
        mode=mode,
        warmup=warmup,
        steps=steps,
        context_length=context_length,
        batch_size=batch_size,
        dtype=dtype,
        annotate_attention=annotate_attention,
        mixed_precision=mixed_precision,
        mixed_dtype=mixed_dtype,
        results_file=results_file,
        memory_profile=memory_profile,
        memory_snapshot_path=memory_snapshot_path,
        annotate_blocks=annotate_blocks,
        checkpoint_block_size=checkpoint_block_size,
        compile=compile,
    )
    print("=" * 60)
    print(result)
