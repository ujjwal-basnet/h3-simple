"""Standalone H3 hybrid INT8 or legacy NF4 experiment with disk offload."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import traceback

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
ROOT = Path(__file__).resolve().parent


def run():
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    from diffsynth.utils.data.audio_video import write_video_audio

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "turbo", "turbo-quality", "selflift", "selflift-quality"], default="smoke")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--vram-limit", type=float, default=10)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--frames", type=int, default=39)
    parser.add_argument("--seed", type=int, default=8143)
    parser.add_argument("--attention-fp16", action="store_true")
    parser.add_argument("--checkpoint", choices=["hybrid", "nf4"], default="hybrid")
    parser.add_argument("--vae-tile-size", type=int, choices=[128,256,384,640,800], default=256)
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()
    if args.frames < 22 or args.frames > 345 or args.frames % 17 != 5:
        parser.error("frames must be 17n+5, between 22 and 345")
    name = f"{args.mode}-{args.dtype}-{time.time_ns()}"
    output = ROOT/"output"
    output.mkdir(exist_ok=True)
    report_path = output/(name+".json")
    report = dict(status="started", mode=args.mode, dtype=args.dtype,
                  vram_limit_gib=args.vram_limit, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), backend="DiffSynth-Studio disk offload",
                  checkpoint=args.checkpoint,
                  selflift=False, diffsynth_commit="974cfa37f27ac55eba3b6d10efa21f876900572d")
    started = time.time()
    def stage(value):
        report.update(stage=value, elapsed_seconds=time.time()-started)
        report_path.write_text(json.dumps(report,indent=2))
        print("STAGE", value, flush=True)
    try:
        dtype = getattr(torch,args.dtype)
        if args.attention_fp16:
            from safe_attention import enable_memory_efficient_attention
            enable_memory_efficient_attention()
        report.update(attention_dtype="float16" if args.attention_fp16 else args.dtype)
        # Full disk staging keeps host memory demand low. T4 computation is
        # explicitly configurable: native BF16 is unavailable on this GPU.
        offload = dict(offload_dtype="disk",offload_device="disk",
                       onload_dtype="disk",onload_device="disk",
                       preparing_dtype="disk",preparing_device="disk",
                       computation_device="cuda")
        stage("load_models")
        names=["minimax-h3-fl2va-nf4.safetensors","minimax-h3-text-encoder-nf4.safetensors",
               "video_vae_nf4.safetensors","audio_vae_nf4.safetensors"]
        paths=[ROOT/"models/nf4"/name for name in names]
        if args.checkpoint == "hybrid":
            from hybrid_checkpoint import PATH, REPO, REVISION, SHA256
            receipt=json.loads((ROOT/"hybrid-download.json").read_text())
            if receipt.get("sha256") != SHA256 or receipt.get("status") != "verified":
                raise ValueError("Run hybrid_checkpoint.py to verify the hybrid weights first")
            paths[0]=PATH
            report.update(checkpoint_repo=REPO, checkpoint_revision=REVISION,
                          checkpoint_sha256=SHA256)
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=dtype,device="cuda",
            model_configs=[ModelConfig(path=str(path),**offload,
                computation_dtype=dtype if index == 0 else torch.float32)
                for index,path in enumerate(paths)],
            processor_config=ModelConfig(path=str(ROOT/"models/h3/FL2VA/processor")),
            vram_limit=args.vram_limit)
        raw_video_decode=pipe.video_vae.decode_video
        raw_audio_decode=pipe.audio_vae.decode_audio
        def video_decode(latents, **kwargs):
            kwargs["dtype"]=torch.float32
            kwargs["tile_size"]=args.vae_tile_size
            if args.save_latents:
                torch.save(latents.detach().cpu(),output/(name+"-video-latents.pt"))
            return raw_video_decode(latents.float(), **kwargs)
        def audio_decode(latents, **kwargs):
            kwargs["dtype"]=torch.float32
            return raw_audio_decode(latents.float(), **kwargs)
        pipe.video_vae.decode_video=video_decode
        pipe.audio_vae.decode_audio=audio_decode
        turbo_file="minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors" if args.mode in ("turbo-quality", "selflift-quality") else "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
        if args.mode!="smoke":
            stage("load_turbo")
            if args.checkpoint == "hybrid":
                from hybrid_lora import HybridTurboLoader
                pipe.lora_loader=HybridTurboLoader
            pipe.load_lora(pipe.dit,ModelConfig(path=str(ROOT/"models/turbo"/turbo_file)))
            report.update(turbo_file=turbo_file)
        width,height,frames,steps=(320,192,22,2) if args.mode=="smoke" else (640,384,39,4)
        transition=2
        if args.mode=="turbo-quality":
            steps=8
        prompt="A ceramic teapot pours warm tea into a cup beside a rainy window. One continuous close shot, natural motion, soft rain ambience, no speech, no text."
        if args.mode=="selflift-quality":
            width,height,frames,steps,transition=800,480,39,8,6
            prompt="A cinematic close shot of a crystal-clear forest stream flowing over dark moss-covered rocks. Crisp wet stone textures, delicate green fern leaves and soft morning sunlight. A locked steady camera shows continuous gentle water motion. Photorealistic natural colors. Quiet flowing water ambience, no speech, no music, no text."
        if args.mode in ("selflift", "selflift-quality"):
            from selflift import attach_selflift
            attach_selflift(pipe, height=height, width=width, transition_step=transition,
                            vae_tile_size=args.vae_tile_size)
            report.update(selflift=True, transition_step=transition, rho=0.4,
                          correction="adaptive_selected_region_eq8", wmin=0.5, wmax=1.0,
                          target_width=width, target_height=height)
            width,height=(640,384) if args.mode=="selflift-quality" else (320,192)
        else:
            original_prediction=pipe.cfg_guided_model_fn
            def checked_prediction(*call_args, **kwargs):
                prediction=original_prediction(*call_args, **kwargs)
                if not all(bool(torch.isfinite(value).all()) for value in prediction):
                    raise ValueError("Denoiser produced non-finite values; precision trial stopped")
                return prediction
            pipe.cfg_guided_model_fn=checked_prediction
        frames=22 if args.mode=="smoke" else args.frames
        if args.prompt_file is not None:
            prompt=args.prompt_file.read_text().strip()
        report.update(width=width,height=height,frames=frames,steps=steps,
                      vae_tile_size=args.vae_tile_size,save_latents=args.save_latents,
                      turbo=args.mode!="smoke",quality_test=args.mode!="smoke",prompt=prompt,
                      seed=args.seed,text_encoder_computation_dtype="float32",
                      video_vae_computation_dtype="float32",audio_vae_computation_dtype="float32")
        stage("generate")
        video,audio=pipe(prompt=prompt,
                         width=width,height=height,num_frames=frames,num_inference_steps=steps,
                         seed=args.seed,flow_shift=6.0 if args.mode!="smoke" else 12.0,
                         cfg_scale=1,tiled=True,tile_size=args.vae_tile_size,tile_overlap=32)
        if not torch.isfinite(torch.as_tensor(audio)).all():
            raise ValueError("Generated audio contains non-finite samples")
        report["selflift_diagnostics"]=getattr(pipe,"selflift_diagnostics",None)
        stage("export")
        path=output/(name+".mp4")
        write_video_audio(video=video,audio=audio,output_path=str(path),fps=24,audio_sample_rate=32000)
        report.update(status="success",video_file=path.name,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_resident_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        stage("complete")
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error),
                      traceback=traceback.format_exc(),elapsed_seconds=time.time()-started,
                      peak_resident_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        report_path.write_text(json.dumps(report,indent=2))
        traceback.print_exc()
        raise


if __name__=="__main__":
    run()
