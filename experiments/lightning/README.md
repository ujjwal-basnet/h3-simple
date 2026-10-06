# Lightning low-memory experiment

Standalone MiniMax-H3 with DiffSynth-Studio, pre-quantized NF4 weights and disk offloading. This is an experiment, not a verified 12 GB GPU recipe. `selflift.py` implements an experimental text-to-video SelfLift-zero adaptation using paired latent/pixel lifts, disagreement correction and re-noising. It keeps the audio trajectory and rebuilds the high-resolution packed layout.

The current Studio folder is `/teamspace/studios/this_studio/h3-lowram-experiment-20261006`. Open `render.log` to watch progress and `output/` for completed videos and receipts. `memory-observation.json` records sampled worker RAM and device memory.

Run from this folder using the isolated environment:

```bash
.venv/bin/python download_models.py
.venv/bin/python monitor_run.py --mode smoke --dtype float32 --vram-limit 10
.venv/bin/python monitor_run.py --mode turbo --dtype float32 --vram-limit 10
.venv/bin/python monitor_run.py --mode selflift --dtype float32 --vram-limit 10
.venv/bin/python render_story.py
```

Run one experiment at a time. The smoke run uses 320×192, 22 frames and two steps to test loading and inference; it is not a quality demonstration. Turbo uses 640×384, 39 frames and four steps. The 10 GiB setting is a backend memory budget, not proof of operation on a physical 12 GB GPU. Disk offloading trades speed for lower memory use.

SelfLift uses 39 frames and four Turbo steps, starting at 320×192 and lifting to 640×384 before the third step. This experimental adapter supports text-to-video with CFG=1; it is not an image-conditioning implementation or a quality guarantee. See [measured results](results/README.md) for completed runs.

The corrected `selflift-quality` preset uses the matching eight-step adapter, adaptive correction weights from 0.5 to 1.0, rho=0.4 and a 640×384 → 800×480 transition before step seven. `render_story.py` downloads that adapter and generates three 124-frame text-to-video scenes with FP16 rendering and FP32 text-encoder computation, then joins and trims them into a 15-second film. Its storyboard is [prompts/storm-guardian.yaml](prompts/storm-guardian.yaml). Watch `story.log` and `story-job.json` for progress. This longer FP16 preset is being tested separately from the earlier FP32 measurements; see [paper review](SELF_LIFT_REVIEW.md).

Setup in a new environment with uv:

```bash
uv sync --locked
uv run download_models.py
uv run monitor_run.py --mode turbo --dtype float32 --vram-limit 10
```

Credits: [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [DiffSynth-Studio NF4](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4), [lightx2v Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo), and [SelfLift research](https://arxiv.org/abs/2609.02036). Model and adapter licenses apply separately. These scripts do not claim ownership of the models, upstream inference engine or SelfLift research.
