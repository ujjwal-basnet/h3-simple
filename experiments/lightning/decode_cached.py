"""Re-decode saved video latents without repeating text encoding or denoising."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
ROOT=Path(__file__).resolve().parent


def run():
    import imageio_ffmpeg
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    parser=argparse.ArgumentParser()
    parser.add_argument("latents", type=Path)
    parser.add_argument("--tile-size", type=int, choices=[128,256,384,640,800], default=384)
    args=parser.parse_args()
    target=args.latents.with_name(args.latents.stem+f"-decode-tile{args.tile_size}")
    report=dict(status="started", source_latents=str(args.latents), tile_size=args.tile_size,
                audio="none: decoder-only comparison", denoising_steps=0)
    def save():
        target.with_suffix(".json").write_text(json.dumps(report,indent=2))
    save()
    started=time.time()
    try:
        config=dict(offload_dtype="disk", offload_device="disk", onload_dtype="disk",
                    onload_device="disk", preparing_dtype="disk", preparing_device="disk",
                    computation_dtype=torch.float32, computation_device="cuda")
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.float32,device="cuda",
            model_configs=[ModelConfig(path=str(ROOT/"models/nf4/video_vae_nf4.safetensors"),**config)],
            vram_limit=10)
        pipe.load_models_to_device(["video_vae"])
        latents=torch.load(args.latents,map_location="cpu",weights_only=True).to("cuda",torch.float32)
        tick=time.time()
        with torch.inference_mode():
            pixels=pipe.video_vae.decode_video(latents,dtype=torch.float32,tile_size=args.tile_size,tile_overlap=32)
        torch.cuda.synchronize()
        report["decode_seconds"]=time.time()-tick
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError("Decoder produced non-finite pixels")
        height,width=pixels.shape[-2:]
        writer=imageio_ffmpeg.write_frames(str(target.with_suffix(".mp4")),(width,height),
                                           fps=24,codec="libx264",quality=8)
        writer.send(None)
        try:
            for frame in pixels[0].permute(1,2,3,0):
                writer.send((frame.clamp(0,1)*255).to(torch.uint8).cpu().contiguous().numpy())
        finally:
            writer.close()
        report.update(status="success",width=width,height=height,frames=pixels.shape[2],
                      elapsed_seconds=time.time()-started,
                      peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        save()
        print(json.dumps(report,indent=2),flush=True)
    except Exception as error:
        report.update(status="failed",error=str(error),elapsed_seconds=time.time()-started)
        save()
        raise


if __name__ == "__main__":
    run()
