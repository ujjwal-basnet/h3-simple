"""Plain Python H3/Turbo/SelfLift orchestration; no ComfyUI imports.

SelfLift-zero equations are independently implemented from the paper. The H3
adaptation is experimental. Full model inference requires a suitable runtime.
"""
from settings import VideoSettings
from pathlib import Path
import inspect
import json
import shutil
import subprocess
import time

DIFFUSERS_REVISION = "899c9f3fd0e64f3206781c00c71c28249eff9ca5"
MODEL_ID = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
TURBO_ID = "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI"
TURBO_REVISION = "bb2bc497cbaca89dadd0bcf1856eed4f8275be20"
TURBO_FILE = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"



def inspect_runtime(cache_dir="/content/h3-clean-models", require_capacity=False):
    """Check before fetching large original-format checkpoints."""
    import torch
    ram = int(next(x for x in Path("/proc/meminfo").read_text().splitlines()
                   if x.startswith("MemTotal:")).split()[1]) / 2**20
    path = Path(cache_dir)
    while not path.exists():
        path = path.parent
    report = dict(system_ram_gib=round(ram, 2),
                  free_disk_gib=round(shutil.disk_usage(path).free / 2**30, 2),
                  cuda_available=torch.cuda.is_available())
    if report["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        report.update(gpu=props.name, gpu_memory_gib=round(props.total_memory / 2**30, 2),
                      native_bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False))
    print(json.dumps(report, indent=2))
    if require_capacity:
        if not report["cuda_available"]:
            raise RuntimeError("Full H3 rendering requires a CUDA GPU.")
        if ram < 80:
            raise RuntimeError("Stopped before model download: this int8 CPU-offload recipe needs approximately "
                               "75 GiB of host RAM; use at least 80 GiB. The existing 13 GiB T4 runtime is insufficient.")
        if report["free_disk_gib"] < 140:
            raise RuntimeError("Allow at least 140 GiB free for the original checkpoint download/cache. "
                               "Existing Comfy-format weights are not reused by this loader.")
    return report


def load_models(cache_dir="/content/h3-clean-models", device="cuda"):
    """Official Diffusers int8 + CPU block-offload recipe, one transformer only."""
    inspect_runtime(cache_dir, require_capacity=True)
    import torch
    from diffusers import ModularPipeline, MiniMaxH3Transformer3DModel, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as EncoderQuantization
    from torchao.quantization import Int8WeightOnlyConfig

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This conservative loader targets native BF16 GPUs (for example A100/L4). "
                           "Standalone T4 loading is not validated; the working Comfy notebook remains available.")
    dtype = torch.bfloat16
    common = dict(revision=MODEL_REVISION, cache_dir=cache_dir, dtype=dtype)
    pipe = ModularPipeline.from_pretrained(MODEL_ID, workflow="fl2va",
                                          revision=MODEL_REVISION, cache_dir=cache_dir)
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", **common,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
            "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
            "token_refiner", "norm_out", "proj_out", "audio_proj_out"]), low_cpu_mem_usage=False)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", **common,
        quantization_config=EncoderQuantization(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
            "model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head"]))
    pipe.update_components(transformer=transformer, text_encoder=encoder)
    pipe.load_components(dtype=dtype, revision=MODEL_REVISION, cache_dir=cache_dir)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    return pipe


def configure_memory(pipe, device="cuda"):
    """Install memory hooks AFTER loading adapters, so LoRA modules are included."""
    import torch
    from diffusers.hooks import apply_group_offloading
    # Streaming is off to avoid doubling host memory. This is a conservative recipe.
    offload = dict(onload_device=torch.device(device), offload_device=torch.device("cpu"), use_stream=False)
    pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
    apply_group_offloading(pipe.vae, offload_type="leaf_level", **offload)
    pipe.vae.enable_tiling()
    pipe.audio_vae.to(device)
    return pipe


def apply_lora(pipe, source=TURBO_ID, filename=TURBO_FILE, strength=1.0):
    """Use Diffusers' H3 converter; unsupported adapter keys must raise."""
    kwargs = dict(adapter_name="turbo")
    if filename:
        kwargs["weight_name"] = filename
    if source == TURBO_ID:
        kwargs["revision"] = TURBO_REVISION
    pipe.load_lora_weights(source, **kwargs)
    pipe.set_adapters("turbo", adapter_weights=float(strength))
    return pipe


def artifact_correct(direct, pixel_anchor, rho=0.6):
    """Correct the top-rho latent locations by paired-lift disagreement (weight 1)."""
    import torch
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1].")
    if direct.shape != pixel_anchor.shape:
        raise ValueError("The two lifts must have identical shapes.")
    if rho == 0:
        return direct
    if rho == 1:
        return pixel_anchor
    delta = pixel_anchor.float() - direct.float()
    risk = delta.abs().mean(dim=1)
    threshold = torch.quantile(risk.flatten(1), 1-rho, dim=1)
    threshold = threshold.reshape((-1,) + (1,) * (risk.ndim-1))
    return direct.float() + (risk >= threshold).unsqueeze(1) * delta


def selflift_transition(clean_low, pixel_anchor_fn, target_hw, sigma_resume, noise, rho=0.6):
    """SelfLift-zero: paired clean lifts → artifact correction → re-noising.

    pixel_anchor_fn decodes, resizes and re-encodes in the same VAE latent space.
    noise is fresh unit Gaussian noise at the target shape. H3 uses sigma=1-t.
    """
    import torch
    import torch.nn.functional as F
    direct = F.interpolate(clean_low.float(), size=(clean_low.shape[2], *target_hw), mode="nearest")
    anchor = pixel_anchor_fn(clean_low, target_hw)
    corrected = artifact_correct(direct, anchor, rho)
    if not 0 <= sigma_resume <= 1 or noise.shape != corrected.shape:
        raise ValueError("Invalid resume sigma or target noise shape.")
    return (1-sigma_resume)*corrected.to(noise.device) + sigma_resume*noise


def pixel_vae_anchor(pipe, clean_low, target_hw):
    """H3 VAE round trip with both latent and ImageNet pixel normalization."""
    import torch
    import torch.nn.functional as F
    device = pipe._execution_device
    dtype = pipe.vae.dtype
    shape = (1, -1, 1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=device).view(shape)
    std = torch.tensor(pipe.vae.config.latents_std, device=device).view(shape)
    pixels = pipe.vae.decode((clean_low.to(device)*std+mean).to(dtype), return_dict=False)[0]
    pm = torch.tensor(pipe.pixel_mean, device=device).view(shape)
    ps = torch.tensor(pipe.pixel_std, device=device).view(shape)
    pixels = (pixels.float()*ps+pm).clamp(0, 1)
    b, c, t, h, w = pixels.shape
    frames = pixels.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
    frames = F.interpolate(frames, size=(target_hw[0]*16, target_hw[1]*16),
                           mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    pixels = frames.reshape(b, t, c, *frames.shape[-2:]).permute(0, 2, 1, 3, 4)
    posterior = pipe.vae.encode(((pixels-pm)/ps).to(dtype), return_dict=False)[0]
    # Deterministic anchor: posterior mean, not a fresh stochastic posterior draw.
    return ((posterior.mode().float()-mean)/std).to(clean_low.device)


def _prepare(pipe, settings, size, generator, conditioning=None):
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from PIL import Image
    h, w = size
    values = dict(prompt=settings.prompt, height=h, width=w, num_frames=settings.frames,
                  generator=generator, num_inference_steps=settings.steps+1, output_type="pil")
    if settings.image_path:
        with Image.open(settings.image_path) as im:
            values["image"] = im.convert("RGB")
    state = PipelineState(values=values)
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    top = MiniMaxH3Blocks().get_workflow("fl2va" if settings.image_path else "t2va").sub_blocks
    if "before_encode" in top:
        pipe, state = top["before_encode"](pipe, state)
    if conditioning is None:
        pipe, state = top["text_encoder"](pipe, state)
    else:
        for key, value in conditioning.items():
            state.set(key, value)
    if "vae_encoder" in top:
        pipe, state = top["vae_encoder"](pipe, state)
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
        MiniMaxH3FL2VACoreDenoiseStep, MiniMaxH3CoreDenoiseStep)
    core = MiniMaxH3FL2VACoreDenoiseStep() if settings.image_path else MiniMaxH3CoreDenoiseStep()
    for name, block in core.sub_blocks.items():
        if name == "denoise":
            break
        pipe, state = block(pipe, state)
    return state, core


def _unpack(pipe, state, rows):
    p = pipe.patch_size
    c = pipe.vae_latent_channels
    t, h, w = (state.get(k) for k in ("num_latent_frames", "latent_height", "latent_width"))
    return rows.reshape(1,t//p[0],h//p[1],w//p[2],c,*p).permute(0,4,1,5,2,6,3,7).reshape(1,c,t,h,w)


def _predict(pipe, state, i):
    import torch
    unique, indices = state.get("row_timestep_plan")[i]
    fields = {k:v for k,v in state.get_by_kwargs("denoiser_input_fields").items()
              if k in inspect.signature(pipe.transformer.forward).parameters}
    with torch.no_grad():
        return pipe.transformer(hidden_states=state.get("latents")[None],
            audio_hidden_states=state.get("audio_latents")[None],
            encoder_hidden_states=state.get("prompt_embeds"), timestep=unique,
            timestep_indices=indices, return_dict=False, **fields)


def generate_video(pipe, settings, output_dir="/content/h3-clean/output", use_selflift=True):
    """Joint video/audio Euler loop, optional progressive-resolution SelfLift-zero.

    Uses the official native H3 schedule; not the previous Comfy Beta schedule.
    No high-resolution denoiser tiling or TST is claimed in this implementation.
    """
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    settings = VideoSettings.model_validate(settings)
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    started = time.time()
    size = settings.low_size if use_selflift else (settings.height, settings.width)
    state, core = _prepare(pipe, settings, size, generator)
    with torch.no_grad():
        for i in range(settings.steps):
            velocity, audio_velocity = _predict(pipe, state, i)
            n = state.get("num_condition_video_rows", 0)
            an = state.get("num_condition_audio_rows", 0)
            rows = state.get("latents")
            timestep = state.get("timesteps")[i]
            clean_rows = rows[n:].float() + (1-float(timestep))*velocity[0,n:].float()
            rows[n:] = pipe.scheduler.step(velocity[0,n:].float(), timestep, rows[n:], return_dict=False)[0]
            arows = state.get("audio_latents")
            arows[an:] = pipe.audio_scheduler.step(audio_velocity[0,an:].float(),
                state.get("audio_timesteps")[i], arows[an:], return_dict=False)[0]
            print(f"Step {i+1}/{settings.steps}: {size[1]}×{size[0]}", flush=True)
            if use_selflift and i+1 == settings.transition_step:
                clean_low = _unpack(pipe, state, clean_rows)
                audio = arows.detach().clone()
                conditioning = state.get(["prompt_embeds", "text_token_tags"])
                target = (settings.height//16, settings.width//16)
                anchor = pixel_vae_anchor(pipe, clean_low, target)
                size = (settings.height, settings.width)
                high, core = _prepare(pipe, settings, size, generator, conditioning=conditioning)
                hn = high.get("num_condition_video_rows", 0)
                noise = _unpack(pipe, high, high.get("latents")[hn:])
                sigma = float(pipe.scheduler.sigmas[i+1])
                lifted = selflift_transition(clean_low, lambda z, hw: anchor, target, sigma, noise, settings.rho)
                high.get("latents")[hn:] = patchify_video_latents(lifted, pipe.patch_size)
                high.set("audio_latents", audio)
                # _prepare resets schedules. Resume both at the next global step.
                pipe.scheduler.set_begin_index(i+1)
                pipe.audio_scheduler.set_begin_index(i+1)
                state = high
        pipe, state = core.sub_blocks["after_denoise"](pipe, state)
        from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3DecodeStep
        pipe, state = MiniMaxH3DecodeStep()(pipe, state)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"h3-{settings.seed}-{time.time_ns()}.mp4"
    export_video(state.get("videos")[0], state.get("audio")[0], state.get("sampling_rate"), path, settings)
    receipt = dict(status="success", backend="PyTorch + Modular Diffusers; no ComfyUI",
                   settings=settings.model_dump(mode="json"), frames=settings.frames, selflift=use_selflift,
                   elapsed_seconds=time.time()-started, diffusers_revision=DIFFUSERS_REVISION,
                   model_revision=MODEL_REVISION, turbo_revision=TURBO_REVISION)
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
    return path


def export_video(frames, audio, sample_rate, path, settings):
    from diffusers.utils.export_utils import encode_video
    if settings.export_width or settings.export_height:
        w = settings.export_width or settings.width
        h = settings.export_height or settings.height
        left, top = (settings.width-w)//2, (settings.height-h)//2
        frames = [im.crop((left,top,left+w,top+h)) for im in frames]
    encode_video(frames, fps=24, output_path=str(path), audio=audio,
                 audio_sample_rate=sample_rate)
    import imageio_ffmpeg
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
