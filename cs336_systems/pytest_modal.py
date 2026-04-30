import subprocess
import sys

import modal

from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image


@app.function(
    image=build_image(include_tests=True),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    cpu=4,
    memory=16384,
    timeout=86400,
)
def run_pytests(pytest_args: list[str] | None = None) -> None:
    import subprocess

    result = subprocess.run(
        ["uv", "run", "pytest"] + (pytest_args or []), check=False, cwd="/root"
    )  # SHOULD BE FALSE
    if result.returncode != 0:
        print("modal pytest exited with code", result.returncode)


@app.local_entrypoint()
def modal_main(args: str = "") -> None:
    pytest_args = args.split() if args else []
    run_pytests.remote(pytest_args)


if __name__ == "__main__":
    print(
        'uv run modal run cs336_systems/pytest_modal.py --args "-k test_flash_forward_pass_triton -x -v"'
    )
