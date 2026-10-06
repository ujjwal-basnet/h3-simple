# Lightning low-memory experiment

Standalone MiniMax-H3 with DiffSynth-Studio and disk offloading. The default diffusion checkpoint is the user-selected hybrid INT8 model; earlier measured runs used NF4. This is an experiment, not a verified 12 GB GPU recipe. `selflift.py` implements an experimental text-to-video SelfLift-zero adaptation using paired latent/pixel lifts, disagreement correction and re-noising. It keeps the audio trajectory and rebuilds the high-resolution packed layout.

The current Studio folder is `/teamspace/studios/this_studio/h3-lowram-experiment-20261006`. Open `render.log` to watch progress and `output/` for completed videos and receipts. `memory-observation.json` records sampled worker RAM and device memory.

Run from this folder using the isolated environment:

```bash
.venv/bin/python download_models.py
.venv/bin/python hybrid_checkpoint.py
.venv/bin/python monitor_run.py --mode smoke --dtype float32 --vram-limit 10
.venv/bin/python monitor_run.py --mode turbo --dtype float32 --vram-limit 10
.venv/bin/python monitor_run.py --mode selflift --dtype float32 --vram-limit 10
.venv/bin/python render_story.py
```

Run one experiment at a time. The smoke run uses 320×192, 22 frames and two steps to test loading and inference; it is not a quality demonstration. Turbo uses 640×384, 39 frames and four steps. The 10 GiB setting is a backend memory budget, not proof of operation on a physical 12 GB GPU. Disk offloading trades speed for lower memory use.

SelfLift uses 39 frames and four Turbo steps, starting at 320×192 and lifting to 640×384 before the third step. This experimental adapter supports text-to-video with CFG=1; it is not an image-conditioning implementation or a quality guarantee. See [measured results](results/README.md) for completed runs.

The corrected `selflift-quality` preset uses the matching eight-step adapter, adaptive correction weights from 0.5 to 1.0, rho=0.4 and a 640×384 → 800×480 transition before step seven. `render_story.py` downloads that adapter and generates three 124-frame text-to-video scenes, then joins and trims them into a 15-second film. Its storyboard is [prompts/storm-guardian.yaml](prompts/storm-guardian.yaml). Watch `story.log` and `story-job.json` for progress; see [paper review](SELF_LIFT_REVIEW.md).

The longer preset uses FP32 model/latent/VAE arithmetic with the `--attention-fp16` path for H3's attention operation. It casts normalized queries and keys to FP16 and scales values before casting, restoring that scale in FP32. This reduces attention memory while keeping potentially large model activations outside FP16. A full FP16 denoiser trial produced non-finite predictions on the tested T4 and was stopped; use the story renderer's FP32 settings. This longer preset is being tested separately from the earlier small-sample measurements.

Setup in a new environment with uv:

```bash
uv sync --locked
uv run download_models.py
uv run hybrid_checkpoint.py
uv run monitor_run.py --mode turbo --dtype float32 --vram-limit 10
```

Credits: [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [DiffSynth-Studio NF4](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4), [lightx2v Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo), and [SelfLift research](https://arxiv.org/abs/2609.02036). Model and adapter licenses apply separately. These scripts do not claim ownership of the models, upstream inference engine or SelfLift research.
# Selected hybrid checkpoint (2026-10-06)

The runner now defaults to the user's selected hybrid INT8 checkpoint. Install with `uv sync --locked`, run `uv run python hybrid_checkpoint.py`, and then use the existing runner. Use `--checkpoint nf4` to reproduce the older NF4 experiments. This is a standalone Python backend; `comfy-kitchen` is a quantization library, not a ComfyUI runtime.

See [dated findings](../../findings/2026-10-06_h3_hybrid_checkpoint.md) for the pinned source, compatibility evidence, and why this is a preferred candidate rather than a proven best checkpoint. Existing sample results were produced with NF4 and are not hybrid measurements.

VAE tile size now defaults to 256 after a generated-video trial reduced visible grid artifacts and took 359 seconds versus the earlier 392-second baseline. Use `--save-latents` to enable decoder comparisons without repeating diffusion. `decode_cached.py output/NAME-video-latents.pt --tile-size 384` produces a silent comparison clip. `diagnose_vae.py` compares decoder tile sizes on identical synthetic-image latents. See [parallel research findings](../../findings/2026-10-06_parallel_colab_lightning.md). Larger tiles use more memory; synthetic timings are not full-video benchmarks.
