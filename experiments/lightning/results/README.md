# Standalone Lightning results

Measured on a Tesla T4 reporting 15 GiB GPU memory, with roughly 16 GB host RAM. The backend is DiffSynth-Studio with pinned pre-quantized NF4 weights, disk offloading and float32 computation. No ComfyUI dependency. The initial three NF4 runs used the same tea-by-a-rainy-window prompt and seed 8143. These are short experiments, not production-quality or 1080p demonstrations.

| Run | Output | Peak process RAM | Sampled device memory | Runtime |
|---|---|---:|---:|---:|
| [Smoke](smoke.mp4) | 320×192, 22 frames, 2 steps | 5.08 GiB | 4.49 GiB | 178 s |
| [Turbo](turbo.mp4) | 640×384, 39 frames, 4 steps | 5.11 GiB | 9.41 GiB | 427 s |
| [Turbo + experimental SelfLift](selflift.mp4) | 320×192 → 640×384, 39 frames, 4 steps | 5.11 GiB | 9.41 GiB | 453 s |

The two 39-frame clips last 1.625 seconds at 24 fps and include stereo audio. Full media decoding and finite, non-silent decoded audio are checked in the `*-media-check.json` files. Per-run receipts are in `*-report.json`. Process RAM comes from `ru_maxrss`; device memory is sampled every three seconds and can miss brief peaks. Receipts also record PyTorch's maximum allocated GPU memory, which excludes some CUDA/driver allocations.

The smoke output is soft and under-denoised. Turbo gives a clearer result. SelfLift completed the paired VAE/latent lift, rebuilt the high-resolution packed layout and resumed sampling while preserving the audio trajectory; it is an experimental adaptation of SelfLift-zero, not a validated quality improvement. Its extra VAE pass made this run about 6% slower than Turbo. The different starting canvas changes the initial video noise, so this is not a controlled quality comparison.

**Legacy correction:** the SelfLift sample in this table used full replacement at selected locations rather than Eq. 8's adaptive weights. The code has since been corrected after a full paper review. These files are retained as the original measured experiment; see [review notes](../SELF_LIFT_REVIEW.md).

The 10 GiB backend budget is not a hard device-memory cap. These results are encouraging for lower-memory use, but do not verify operation on a physical 12 GB GPU or native 1080p generation. The original Diffusers loader and the older Comfy-backed samples are separate implementations.
# Hybrid checkpoint and decoder improvement — 2026-10-06

[Updated video](hybrid-tile256.mp4) uses the selected hybrid b25–49 INT8 checkpoint, LightX2V four-step Turbo, FP32 computation, guarded FP16 attention, and 256-pixel VAE tiles. Output: 640×384, 39 frames, 24 FPS, approximately 1.625 seconds. Runtime was 359.10 seconds and sampled device peak was 9.41 GiB. SelfLift is disabled for this baseline.

The earlier hybrid trial with 128-pixel tiles took 391.93 seconds and showed grid artifacts. Inspected frames from this updated clip are much cleaner. Full-file FFmpeg decoding passed. This short simple-action clip does not establish quality or timing for a longer combat sequence.

See [receipt](hybrid-tile256-report.json), [preview](hybrid-tile256-preview.png), and [dated parallel research findings](../../../findings/2026-10-06_parallel_colab_lightning.md). Earlier NF4 samples retain their original provenance below.


## Original storm guardian controls

These text-to-video controls use the selected hybrid checkpoint and original guardian prompt, seed 9175, FP32 computation, guarded FP16 attention, and 256-pixel VAE tiles. Both are readable in inspected middle frames and pass full FFmpeg decoding. Neither uses SelfLift.

| Sample | Native output | Runtime | Turbo metadata scale |
| --- | --- | ---: | ---: |
| [Four-step guardian](storm-guardian-turbo-control.mp4) | 640×384, 39 frames, 1.625 seconds | 367.93 seconds | 1.0 |
| [Corrected eight-step guardian](storm-guardian-eightstep-scaled.mp4) | 640×384, 22 frames, about 0.917 seconds | 572.92 seconds | 0.0625 |

The clips differ in frame count and adapter, so their runtime comparison does not isolate step count. Earlier unscaled eight-step trials produced mostly noise; the loader now applies alpha/rank from the adapter metadata. A corrected SelfLift trial is being validated separately. These controls do not establish 15-second continuity, 1080p, or operation on a physical 12 GB GPU.

[Corrected SelfLift validation](storm-guardian-selflift-scaled.mp4): 22 frames, 800×480, eight evaluations with transition before step seven, runtime 648.83 seconds. The inspected middle frame is readable, and full decoding passes. This short test does not establish long-video continuity or superiority over a native 800×480 baseline. Its receipt includes transition statistics and the applied adapter scale.
