"""Compare decoder tile sizes on identical latents, without a denoiser run."""
import json
import os
from pathlib import Path
import time

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
ROOT = Path(__file__).resolve().parent


def run():
    import torch
    from PIL import Image
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    output = ROOT / "output/vae-diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = dict(offload_dtype="disk", offload_device="disk", onload_dtype="disk",
                  onload_device="disk", preparing_dtype="disk", preparing_device="disk",
                  computation_dtype=torch.float32, computation_device="cuda")
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.float32, device="cuda",
        model_configs=[ModelConfig(path=str(ROOT / "models/nf4/video_vae_nf4.safetensors"), **config)],
        vram_limit=10)
    pipe.load_models_to_device(["video_vae"])
    height, width = 192, 320
    y, x = torch.meshgrid(torch.linspace(0, 1, height, device="cuda"),
                          torch.linspace(0, 1, width, device="cuda"), indexing="ij")
    pixels = torch.stack([x, y, torch.full_like(x, 0.5)]).unsqueeze(0).unsqueeze(2)
    report = dict(status="started", input="smooth RGB gradient", frames=1,
                  width=width, height=height, encoder_tile_size=384, cases=[])
    def save():
        report["elapsed_seconds"] = time.time() - started
        (output / "report.json").write_text(json.dumps(report, indent=2))
    save()
    with torch.inference_mode():
        latents = pipe.video_vae.encode_video(pixels, dtype=torch.float32,
                    process_image=True, tile_size=384, tile_overlap=32)
        for tile_size in [128, 256, 384]:
            tick = time.time()
            decoded = pipe.video_vae.decode_video(latents, dtype=torch.float32,
                        process_image=True, tile_size=tile_size, tile_overlap=32)
            if not bool(torch.isfinite(decoded).all()):
                raise ValueError(f"Non-finite VAE reconstruction at tile size {tile_size}")
            # H3 can decode one image latent into a short temporal group.
            image = decoded[0, :, 0] if decoded.ndim == 5 else decoded[0]
            Image.fromarray((image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(output / f"tile-{tile_size}.png")
            error = float((image - pixels[0, :, 0]).square().mean().sqrt())
            report["cases"].append(dict(tile_size=tile_size, rmse=error,
                                        decoded_shape=list(decoded.shape), seconds=time.time()-tick))
            save()
            print(json.dumps(report["cases"][-1]), flush=True)
    report["status"] = "success"
    save()


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        path=ROOT / "output/vae-diagnostic/report.json"
        report=json.loads(path.read_text()) if path.exists() else {}
        report.update(status="failed", error=str(error))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        raise
