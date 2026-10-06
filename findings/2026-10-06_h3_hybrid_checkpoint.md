# H3 hybrid checkpoint selection — 2026-10-06

Selected at the user's request: `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`.

## Why select it?

The user reports that the supplied workflow produced better-looking video than our standalone NF4 experiment. We confirmed that our Turbo sample contains baked-in grid artifacts and that the two-step smoke sample is visibly poor. These observations justify testing the checkpoint from the better workflow; they do not isolate the checkpoint as the cause.

The [checkpoint author's model card](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/blob/a36feb17fbd1f20ff4bdd509ccd07e2b7b585a38/README.md) describes a tensor-selection merge: FL2VA provides the base, while REF2VA supplies later-block AdaLN modulation weights for blocks 25–49. The intended benefit is reference conditioning with quality closer to FL2VA. The author recommends trying b25–49 first, describes the results as subjective and experimental, and does **not** expect it to beat FL2VA on generation without references. There was no additional training.

Accordingly, “best” here means **the user's preferred checkpoint from a previously better workflow**, not a benchmark-proven winner. In our current text-only story, the merge's reference pathway provides no demonstrated advantage. The supplied graph's reference-image loaders are mode 4 (bypassed), so that JSON alone is not evidence that references contributed to its result.

## Verified artifact and loader compatibility

- Repository: `smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models`.
- Pinned revision: `a36feb17fbd1f20ff4bdd509ccd07e2b7b585a38`.
- File size: 20,970,379,632 bytes, approximately 19.53 GiB.
- Published LFS SHA-256: `a629cfea8d89a071b140c6e1935dc9a23e72de6badc18975a2bb9e6d1423d76d`.
- Remote safetensors header: 932 tensors; 200 INT8 weights, 200 quantization markers, and remaining BF16/FP16/FP32 tensors.
- Examined marker: `int8_tensorwise`, convolutional rotation enabled, group size 256.
- Header keys/shapes produce DiffSynth structural hash `ac9fdcd56900c9a1a5ef21eaacf6cb76`.

The pinned [DiffSynth model registry](https://github.com/modelscope/DiffSynth-Studio/blob/974cfa37f27ac55eba3b6d10efa21f876900572d/diffsynth/configs/model_configs.py) recognizes that hash as `MiniMaxH3DiTComfyPruned` with its `comfy_kitchen_int8_w8a8` backend. The [architecture implementation](https://github.com/modelscope/DiffSynth-Studio/blob/974cfa37f27ac55eba3b6d10efa21f876900572d/diffsynth/models/minimax_h3_dit_comfy.py) handles the checkpoint's QKV ordering and pruned timestep/AdaLN representation. A structural hash proves format recognition, not weight identity; the downloader separately verifies the full SHA-256.

This uses the standalone `comfy-kitchen` quantization library, **not a ComfyUI server or node graph**. It avoids treating INT8 values as ordinary floating-point weights or discarding convolutional-rotation metadata.

## Implemented change

`experiments/lightning/run_experiment.py` now selects this hybrid checkpoint by default. `--checkpoint nf4` preserves the earlier experiment for comparison. `hybrid_checkpoint.py` downloads the pinned file and verifies size and SHA-256 before writing its receipt. `comfy-kitchen` is included in the uv dependency lock.

The diffusion checkpoint is changed. `hybrid_lora.py` also adapts the LightX2V adapter's QKV rows to the checkpoint's contiguous Q/K/V ordering; using the native interleaved ordering would apply the adapter to the wrong output rows. The existing NF4 text encoder and VAEs, LightX2V Turbo adapter, and interpolation-based SelfLift-zero transition remain separate components. This does not reproduce the supplied graph's complete quality recipe, which also names different adapters, a learned latent upscaler, a refinement step, and attention patches.

## Validation and limits

Header inspection and registry compatibility are verified. The full file was downloaded on Lightning and its SHA-256 matched the published value. No hybrid video quality, runtime, RAM, or VRAM result is established by this note. Do not reuse NF4 measurements as hybrid measurements. The visible grid artifacts' root cause remains unresolved; checkpoint quantization, decoding, and other pipeline differences require controlled comparison.

The targeted two-head adapter check passed: Q, K, and V updates occupy the expected contiguous output rows after conversion. Lightning subsequently loaded the exact hybrid file, patched 208 adapter tensors, and reached the generation stage for a 640×384, 39-frame, four-step baseline. This establishes loading and adapter attachment; a completed and visually inspected render is still required.

The previous NF4 story's first scene exported, but an inspected frame is dominated by grid-like noise. The remaining NF4 scenes were stopped after the user selected this checkpoint; the completed clip and measurements are preserved. Longer sampling did not by itself resolve the previous pipeline's quality problem.

Use the same prompt, seed, frames, resolution, scheduler, precision, and decoder when comparing checkpoints. First obtain a clean single-resolution clip, then test SelfLift. Record previews as well as numerical memory/timing data.

The checkpoint is an upstream contribution by its uploader based on MiniMax weights. Our notebook and original prompts can be our own implementation; the checkpoint and research method retain their source attribution.
