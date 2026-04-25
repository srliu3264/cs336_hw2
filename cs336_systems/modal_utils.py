from pathlib import Path, PurePosixPath

import modal

SUNET_ID = "srliu"

if SUNET_ID == "":
    raise NotImplementedError(f"Please set the SUNET_ID in {__file__}")

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"systems-{SUNET_ID}")
user_volume = modal.Volume.from_name(
    f"systems-{SUNET_ID}", create_if_missing=True, version=2
)


def build_image(*, include_tests: bool = False) -> modal.Image:
    # NOTE: pyproject.toml has `cs336-basics = { path = "./cs336-basics", editable = true }`.
    # Modal's uv_sync() runs from /.uv, so we need to ship ./cs336-basics into /.uv/cs336-basics
    # before uv_sync() runs (copy=True bakes it into the image layer at build time).
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("wget", "gzip")
        .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
        .uv_sync()
        .add_local_python_source("cs336_systems")
    )
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


def build_image_nsys(
    *, include_tests: bool = False
) -> modal.Image:  ##  leant from Slack
    """Same as build_image() but adds NVIDIA Nsight Systems for nsys profiling."""
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .run_commands(
            "apt-get update && apt-get install -y wget",
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        )
        .apt_install(
            "libcap2-bin", "libdw1", "cuda-nsight-systems-13-2"
        )  # add nsight-systems
        .add_local_dir(
            "cs336-basics", remote_path="/.uv/cs336-basics", copy=True
        )  # solve cs336-basics dependency
        .uv_sync()
        .add_local_python_source("cs336_systems")
    )
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    f"/root/{DATA_PATH}": user_volume,
}
