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
    timeout=86400,
)
def attention_remote(
    compile: bool = False, results_file: str | None = None, seed: int = 0
):
    from cs336_systems.attention_benchmark import main as attn_main

    # cleaner than sys.argv or subprocess
    args = argparse.Namespace(compile=compile, results_file=results_file, seed=seed)
    try:
        attn_main(args)
    finally:
        if results_file is not None:
            user_volume.commit()


@app.local_entrypoint()
def main(compile: bool = False, results_file: str | None = None, seed: int = 0):
    attention_remote.remote(compile=compile, results_file=results_file, seed=seed)
