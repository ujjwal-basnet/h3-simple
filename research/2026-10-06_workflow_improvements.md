# What improved the H3 workflow — 2026-10-06

The successful result is our **standalone H3 hybrid INT8 + correctly scaled LightX2V eight-step Turbo + experimental SelfLift-zero** pipeline. It runs Python/PyTorch through DiffSynth-Studio without a ComfyUI runtime. It is an independent implementation of the working recipe, not a renamed copy of the supplied BUNNY graph.

## What actually improved the output

| Change | Evidence | Conclusion |
| --- | --- | --- |
| Honor Turbo adapter metadata alpha/rank | Eight-step file: alpha 8, original rank 128; correct scale 0.0625. Our earlier loader used 1.0. An eight-step control without SelfLift also produced noise; corrected Turbo-only and SelfLift clips became readable. | **The main fix for the eight-step noise failure.** The earlier adapter update was 16× too strong. |
| VAE tiles 128 → 256 pixels | Same hybrid checkpoint, teapot prompt, seed and four-step recipe: inspected 256-pixel output reduced grid artifacts. Runtime 391.93 → 359.10 seconds; sampled device peak 9.41 GiB in both runs. Synthetic same-latent decoding also showed grid seams with small tiles. | **A demonstrated decoder improvement**, with about 8.4% lower wall time across two runs. Not a repeated speed benchmark or a cure for every artifact. |
| FP32 model/latent/VAE arithmetic with guarded FP16 attention | Full-model FP16 trials produced non-finite values. The working FP32 recipe exported all three scenes; attention uses bounded FP16 SDPA and restores value scaling in FP32. | **Required for the tested T4 recipe's numerical stability.** It is not a measured visual-quality upgrade against a successful FP16 baseline. |
| Convert native LoRA QKV rows to the hybrid checkpoint's contiguous Q/K/V layout | The selected Comfy-format checkpoint differs from native DiffSynth ordering. A targeted two-head conversion check passed, and 208 adapter tensors were attached. | **Compatibility correction**, not an independently measured aesthetic improvement. |
| Select the b25–49 hybrid INT8 checkpoint | Requested from the user's preferred graph; format and full SHA-256 verified. The corrected pipeline subsequently produced a readable film. | **A validated working checkpoint, not a proven best checkpoint.** No matched checkpoint comparison isolates its advantage. |
| Use a simpler original three-scene prompt | Establishing spear charge, readable spear sweep, then rune activation and protective dome. The rendered scenes visibly follow these broad actions. | A practical prompt choice. Its benefit was not isolated in an A/B test. |

Whole-frame VAE decoding was also tried on cached generated latents. It was faster, but an inspected frame showed more grid/triangular artifacts than the 256-pixel result. Bigger tiles are therefore not automatically better; the successful film retains 256.

## What role did SelfLift play?

SelfLift-zero provided the **640×384 → 800×480 resolution transition**. Before evaluation seven of eight, the implementation estimates the clean latent, builds nearest-neighbor latent and decoded/resized/re-encoded pixel branches, applies selected-region adaptive correction, re-noises at the current sigma and rebuilds H3's packed layout. The audio trajectory continues.

Settings: transition step 6 (before evaluation 7), rho 0.4, correction weights 0.5–1.0, unchanged sampling schedule. This is the experimental H3 adaptation of [SelfLift-zero](https://arxiv.org/abs/2609.02036), not a trained SelfLift-rich model, not a SelfLift LoRA, and not a learned latent upscaler.

**SelfLift was not shown to be the cause of the visual recovery or an overall speed improvement.** The unscaled eight-step adapter failed even without SelfLift. Correcting its scale restored the Turbo-only control first. The subsequent corrected SelfLift control and long scenes established that this transition executes and produces readable output. A matched native 800×480 baseline is still required to measure a SelfLift quality or speed advantage.

## Exact successful recipe

- Diffusion checkpoint: `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`, from `smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models`, revision `a36feb17fbd1f20ff4bdd509ccd07e2b7b585a38`.
- Turbo adapter: `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors`, from `lightx2v/Minimax-h3-Turbo`, revision `3ec17a324ced54151364f24f8b5fb6bf7e26414f`; alpha/rank **8/128 = 0.0625**.
- Companion components: pinned NF4 Qwen text encoder, video VAE and audio VAE; original H3 processor.
- Backend: DiffSynth-Studio commit `974cfa37f27ac55eba3b6d10efa21f876900572d`; PyTorch 2.8.0+cu128; disk offloading; standalone `comfy-kitchen` quantization library.
- Sampling: 8 Euler evaluations, CFG 1, video shift 6, audio shift 3; 124 frames per scene at 24 fps; seeds 9175, 9176, 9177. Transition noise uses the implementation's default seed 9174.
- Precision: FP32 model/latents/text encoder/VAEs; guarded FP16 attention only.
- VAE tiles: 256 pixels, overlap 32.
- Resolution: six lower-resolution evaluations at 640×384, then SelfLift and two at 800×480.
- Three native scenes joined and trimmed to **360 frames / 15.0 seconds of video**. AAC container duration is 15.022 seconds.

The supplied BUNNY graph instead names Turbo v4 plus combat/style adapters, a learned upscaler, extra refinement and TST/SOL patches. We did not reproduce or validate that full graph. Those additions are not evidence for the improvements in our standalone output. Models, inference engine and SelfLift research retain upstream attribution; our prompts and orchestration are our own implementation.

## Measured result and limits

| Measurement | Result |
| --- | ---: |
| Final film | 800×480, 24 fps, 360 frames, generated audio |
| Whole three-scene job | 4410.27 seconds — **73 minutes 30 seconds** |
| Scene worker runtimes | 1417.08 / 1416.44 / 1417.77 seconds |
| Maximum process RAM high-water mark | **5.57 GiB** |
| Maximum sampled device memory | **14.54 GiB** |
| Maximum Torch-allocated / recorded reserved memory | **10.20 / 14.40 GiB** |

Full-file FFmpeg decoding passes; decoded audio is finite and non-silent. Sampled frames across all scenes are readable. Exact character identity, anatomy and full prompt adherence are not guaranteed. Device sampling can miss short peaks. These measurements do not establish native 1080p or operation on a physical 12 GB GPU.

## Evidence

- [Completed film](../experiments/lightning/results/storm-guardian-15s.mp4)
- [Job receipt](../experiments/lightning/results/storm-guardian-story-report.json)
- [Final media check](../experiments/lightning/results/storm-guardian-15s-media-check.json)
- [Corrected Turbo-only control](../experiments/lightning/results/storm-guardian-eightstep-scaled.json)
- [Corrected short SelfLift control](../experiments/lightning/results/storm-guardian-selflift-scaled.json)
- [Decoder trial receipt](../experiments/lightning/results/hybrid-tile256-report.json)
- [Detailed experiment findings](../findings/2026-10-06_parallel_colab_lightning.md)
- [Checkpoint selection findings](../findings/2026-10-06_h3_hybrid_checkpoint.md)

The clean version should preserve this verified recipe and its alpha/rank handling. Quantization, offloading and attention changes need separate measurements; speculative Antigravity optimizations were not applied to the completed film.
