# Parallel Colab research and Lightning experiments — 2026-10-06

The previous Colab session had expired. A new CPU research session, `h3-research-20261006`, was created; no runtime was reset. Lightning remains the T4 inference machine. Antigravity CLI supplied a read-only code audit while Colab examined pinned source and Lightning ran controlled decoder experiments.

## Colab source analysis

The pinned [H3 VAE implementation](https://github.com/modelscope/DiffSynth-Studio/blob/974cfa37f27ac55eba3b6d10efa21f876900572d/diffsynth/models/minimax_h3_video_vae.py) interprets tile size in pixels. Executing its actual `split_tiles` method on Colab produced:

| Output dimensions | 128-pixel tiles | 256-pixel tiles | Whole-frame tile |
| --- | ---: | ---: | ---: |
| 640×384 | 28 | 6 | 1 |
| 800×480 | 40 | 8 | 1 |

These are tile counts, not speed measurements. Larger tiles require more activation memory.

Colab also inspected byte ranges from the pinned hybrid checkpoint. The timestep table's maximum absolute value is 0.468; the sampled block-25 AdaLN linear weight and bias maxima are 11.49 and 7.33. None of these sampled stored values exceed FP16 range. This does not locate or rule out overflow during the actual forward pass, nor inspect every tensor.

## Lightning controlled decoder comparison

The same NF4 VAE encoded a smooth RGB gradient at 320×192 using a whole-frame tile. The identical resulting latent tensor was decoded with three tile sizes. No diffusion checkpoint, prompt, seed, or denoising step was changed between decodes.

| Decoder tile size | Observed decode time | Reconstruction RMSE |
| --- | ---: | ---: |
| 128 | 5.75 seconds | 0.11269 |
| 256 | 2.00 seconds | 0.11248 |
| 384, whole frame | 1.01 seconds | 0.11342 |

This is a single ordered pass; warm-up and caching can affect timing. It is not a rigorous whole-video speed benchmark. One image latent decoded into four frames; the same first frame was inspected in each case. The 128-pixel reconstruction has visible grid seams, whereas larger-tile reconstructions appear much smoother. Similar RMSE values demonstrate why numerical reconstruction error alone does not establish visual quality.

This implicates small-tile decoding in the synthetic test. It does not prove that tiling accounts for every artifact in generated videos, nor validate the NF4 VAE against unquantized weights.

![Same-latent decoder comparison produced on Colab](assets/2026-10-06_decoder-comparison.png)

## Applying the finding

The runner now accepts `--vae-tile-size` and `--save-latents`. A hybrid INT8, FP32, 640×384, 39-frame, four-step trial with 256-pixel VAE tiles completed in **359.10 seconds**, versus **391.93 seconds** for the earlier 128-pixel baseline. The sampled device-memory peak remained **9.41 GiB**. This is approximately **8.4% less wall time** across these two runs; it is not a repeated statistical benchmark.

The inspected generated-video frame no longer shows the earlier large grid artifacts. All other generation settings remained the same except added finite-value checks and latent saving. The 256-pixel setting is now the default; 128 remains available for comparison. SelfLift's pixel-anchor encoder and decoder receive the selected tile size too, but the full longer SelfLift sequence still needs validation with this setting.

Saving generated latents enables subsequent decoder tests without repeating expensive diffusion sampling. `decode_cached.py` completed a 384-pixel decode of the saved real-video latents: **27.65 seconds decoding**, **39.64 seconds including setup/export**, with zero denoising steps. Its output is a silent decoder comparison, not a new diffusion generation or an audio-generation test. A whole-frame 640-pixel decode is the next controlled comparison on Lightning.

The first hybrid FP32 baseline exported in 391.93 seconds but retained visible grid artifacts. Full-model FP16 was separately tried on the hybrid checkpoint and stopped on non-finite attention values. It is not a validated speed optimization.

## Antigravity audit: hypotheses and corrections

Antigravity suggested the shared NF4 VAE, small decoder tiles, and disk staging as likely contributors. Its stronger claims that these were proven causes and that proposed edits would guarantee resolution were not accepted.

Moving all models into FP32 host RAM would exceed the Studio's approximately 15 GiB host memory. Query/key rescaling can preserve attention algebra only for finite inputs; it cannot recover values that have already overflowed upstream. Replacing the NF4 VAE is a reasonable controlled comparison, not an established fix. No untested attention or all-RAM staging patch was applied.

Raw research/audit files are preserved locally under `/home/ujjwal/h3_research/colab`, `/home/ujjwal/h3_research/vae-diagnostic`, and `/home/ujjwal/h3_research/agy-h3-audit-2026-10-06.txt`. Colab research outputs are under `/content/h3-research-20261006`; Lightning diagnostics are under `output/vae-diagnostic` in the experiment folder.
