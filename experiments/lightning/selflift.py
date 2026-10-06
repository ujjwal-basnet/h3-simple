"""Experimental text-to-video SelfLift-zero adaptation for DiffSynth H3.

Independent orchestration of paired latent/pixel lifts and artifact correction.
The upstream VAE handles its own pixel and latent normalization.
"""
import torch
import torch.nn.functional as F
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder


def pixel_anchor(pipe, clean, height, width):
    pipe.load_models_to_device(["video_vae"])
    pixels = pipe.video_vae.decode_video(clean, dtype=pipe.torch_dtype,
                                         tiled=True, tile_size=128, tile_overlap=32)
    batch, channels, frames, _, _ = pixels.shape
    images = pixels.permute(0, 2, 1, 3, 4).reshape(batch*frames, channels, *pixels.shape[-2:])
    images = F.interpolate(images.float(), size=(height, width), mode="bicubic",
                           align_corners=False, antialias=True).clamp(0, 1)
    pixels = images.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    anchor = pipe.video_vae.encode_video(pixels, dtype=pipe.torch_dtype,
                                         tiled=True, tile_size=128, tile_overlap=32)
    pipe.load_models_to_device(pipe.in_iteration_models)
    return anchor


def attach_selflift(pipe, height=384, width=640, transition_step=2, rho=0.6, seed=9174):
    """Install a single-use lift before step 2, keeping the audio trajectory.

    Supports the unconditioned text-to-video experiment with CFG=1 only.
    H3's backend prediction is -velocity and scheduler sigma is timestep/1000.
    """
    original = pipe.cfg_guided_model_fn
    state = {"step": 0}
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    builder = MiniMaxH3Unit_PackedSequenceBuilder()

    def predict(model_fn, cfg_scale, shared, positive, negative, **kwargs):
        if cfg_scale != 1 or any(shared.get(k) is not None for k in
                                ("keyframes", "references", "control_video", "retake_video")):
            raise ValueError("This experiment supports text-to-video with CFG=1 only.")
        sigma = float(kwargs["timestep_video"].item()) / 1000
        if state["step"] == transition_step:
            print("SELFLIFT paired VAE/latent lift", flush=True)
            clean = state["latents"].float() - state["sigma"] * state["prediction"].float()
            anchor = pixel_anchor(pipe, clean, height, width)
            direct = F.interpolate(clean, size=(clean.shape[2], height//16, width//16), mode="nearest")
            if direct.shape != anchor.shape:
                raise ValueError(f"Paired lift shapes differ: {direct.shape} vs {anchor.shape}")
            delta = anchor.float() - direct
            risk = delta.abs().mean(dim=1, keepdim=True)
            threshold = torch.quantile(risk.flatten(1), 1-rho, dim=1).view(-1, 1, 1, 1, 1)
            corrected = direct + (risk >= threshold) * delta
            noise = torch.randn(corrected.shape, generator=generator, device=pipe.device, dtype=pipe.torch_dtype)
            shared["video_latents"] = ((1-sigma)*corrected + sigma*noise).to(pipe.torch_dtype)
            shared.update(height=height, width=width)
            positive.update(builder.process(pipe, prompt_embeds=positive["prompt_embeds"],
                text_token_tags=positive["text_token_tags"],
                video_latents=shared["video_latents"], audio_latents=shared["audio_latents"]))
            if not torch.isfinite(shared["video_latents"]).all():
                raise ValueError("SelfLift produced non-finite latents")
            print("SELFLIFT resume", tuple(shared["video_latents"].shape), "sigma", sigma, flush=True)
        latents = shared["video_latents"].detach().clone()
        prediction = original(model_fn, cfg_scale, shared, positive, negative, **kwargs)
        state.update(step=state["step"]+1, latents=latents,
                     sigma=sigma, prediction=prediction[0].detach())
        return prediction

    pipe.cfg_guided_model_fn = predict
    return pipe
