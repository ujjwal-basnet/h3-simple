"""Download only pinned pre-quantized H3 weights, processor and Turbo adapter."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent
NF4_REV = "363fdc8fbd7ae55f5b9e7fb87cf3d508df28ee0d"
H3_REV = "42ed227ee7df40d41602854ae760620d6eb651fe"
TURBO_REV = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
FILES = [
    ("minimax-h3-fl2va-nf4.safetensors", 17162138303),
    ("minimax-h3-text-encoder-nf4.safetensors", 15324775807),
    ("video_vae_nf4.safetensors", 1613201536),
    ("audio_vae_nf4.safetensors", 284004112),
]


def download_file(item):
    filename, size = item
    print("Downloading", filename, flush=True)
    path = Path(hf_hub_download("DiffSynth-Studio/MiniMax-H3-NF4", filename,
                               revision=NF4_REV, local_dir=ROOT/"models/nf4"))
    if path.stat().st_size != size:
        raise RuntimeError(f"Wrong downloaded size: {filename}")
    print("Ready", filename, flush=True)
    return dict(filename=filename, bytes=size, revision=NF4_REV)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        manifest = list(pool.map(download_file, FILES))
    snapshot_download("MiniMaxAI/MiniMax-H3", revision=H3_REV,
                      allow_patterns=["FL2VA/processor/*"], local_dir=ROOT/"models/h3")
    hf_hub_download("lightx2v/Minimax-h3-Turbo",
                    "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
                    revision=TURBO_REV, local_dir=ROOT/"models/turbo")
    (ROOT/"model-download.json").write_text(json.dumps(dict(status="success", files=manifest,
        processor_revision=H3_REV, turbo_revision=TURBO_REV), indent=2))
    print("DOWNLOAD_COMPLETE", flush=True)
