"""Measured NF4 + disk-offload H3 experiment. No ComfyUI or huge BF16 download."""
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
    parser.add_argument("--mode", choices=["smoke", "turbo", "selflift"], default="smoke")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--vram-limit", type=float, default=10)
    args = parser.parse_args()
    name = f"{args.mode}-{args.dtype}-{time.time_ns()}"
    output = ROOT/"output"
    output.mkdir(exist_ok=True)
    report_path = output/(name+".json")
    report = dict(status="started", mode=args.mode, dtype=args.dtype,
                  vram_limit_gib=args.vram_limit, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), backend="DiffSynth-Studio NF4 disk offload",
                  selflift=False, diffsynth_commit="974cfa37f27ac55eba3b6d10efa21f876900572d")
    started = time.time()
    def stage(value):
        report.update(stage=value, elapsed_seconds=time.time()-started)
        report_path.write_text(json.dumps(report,indent=2))
        print("STAGE", value, flush=True)
    try:
        dtype = getattr(torch,args.dtype)
        # Full disk staging keeps host memory demand low. T4 computation is
        # explicitly configurable: native BF16 is unavailable on this GPU.
        offload = dict(offload_dtype="disk",offload_device="disk",
                       onload_dtype="disk",onload_device="disk",
                       preparing_dtype="disk",preparing_device="disk",
                       computation_dtype=dtype,computation_device="cuda")
        stage("load_models")
        names=["minimax-h3-fl2va-nf4.safetensors","minimax-h3-text-encoder-nf4.safetensors",
               "video_vae_nf4.safetensors","audio_vae_nf4.safetensors"]
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=dtype,device="cuda",
            model_configs=[ModelConfig(path=str(ROOT/"models/nf4"/name),**offload) for name in names],
            processor_config=ModelConfig(path=str(ROOT/"models/h3/FL2VA/processor")),
            vram_limit=args.vram_limit)
        if args.mode in ("turbo", "selflift"):
            stage("load_turbo")
            pipe.load_lora(pipe.dit,ModelConfig(path=str(ROOT/"models/turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")))
        width,height,frames,steps=(320,192,22,2) if args.mode=="smoke" else (640,384,39,4)
        if args.mode=="selflift":
            from selflift import attach_selflift
            attach_selflift(pipe, height=height, width=width)
            report.update(selflift=True, transition_step=2, rho=0.6,
                          target_width=width, target_height=height)
            width,height=320,192
        report.update(width=width,height=height,frames=frames,steps=steps,
                      turbo=args.mode!="smoke",quality_test=args.mode!="smoke")
        stage("generate")
        video,audio=pipe(prompt="A ceramic teapot pours warm tea into a cup beside a rainy window. One continuous close shot, natural motion, soft rain ambience, no speech, no text.",
                         width=width,height=height,num_frames=frames,num_inference_steps=steps,
                         seed=8143,flow_shift=6.0 if args.mode!="smoke" else 12.0,
                         cfg_scale=1,tiled=True,tile_size=128,tile_overlap=32)
        if not torch.isfinite(torch.as_tensor(audio)).all():
            raise ValueError("Generated audio contains non-finite samples")
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
