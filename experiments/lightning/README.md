# Lightning low-memory experiment

Standalone MiniMax-H3 with DiffSynth-Studio, pre-quantized NF4 weights and disk offloading. This is an experiment, not a verified 12 GB GPU recipe. SelfLift is not enabled in this backend yet.

The current Studio folder is `/teamspace/studios/this_studio/h3-lowram-experiment-20261006`. Open `render.log` to watch progress and `output/` for completed videos and receipts. `memory-observation.json` records sampled worker RAM and device memory.

Run from this folder using the isolated environment:

```bash
.venv/bin/python download_models.py
.venv/bin/python monitor_run.py --mode smoke --dtype float32 --vram-limit 10
.venv/bin/python monitor_run.py --mode turbo --dtype float32 --vram-limit 10
```

Run one experiment at a time. The smoke run uses 320×192, 22 frames and two steps to test loading and inference; it is not a quality demonstration. Turbo uses 640×384, 39 frames and four steps. The 10 GiB setting is a backend memory budget, not proof of operation on a physical 12 GB GPU. Disk offloading trades speed for lower memory use.

Setup with uv:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python 'git+https://github.com/modelscope/DiffSynth-Studio.git@974cfa37f27ac55eba3b6d10efa21f876900572d' 'transformers==4.57.6' bitsandbytes 'huggingface-hub<1' soundfile av librosa
```

Credits: [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [DiffSynth-Studio NF4](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4), and [lightx2v Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo). Model and adapter licenses apply separately. These scripts do not claim ownership of the models or upstream inference engine.
