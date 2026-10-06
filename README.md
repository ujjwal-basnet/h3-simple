# Simple H3 + Turbo + SelfLift

Install uv once if needed:

```bash
python3 -m pip install uv
```

In this project folder:

```bash
uv sync --locked
uv run main.py
```

Edit the prompt, photo path and dimensions in `main.py`. `settings.py` contains one Pydantic class; types, ranges and existing photo paths are validated automatically. Only relationships between settings need explicit checks. `pyproject.toml` lists the Python packages; `uv.lock` records their exact resolved versions. uv creates a separate `.venv` and does not replace your existing Comfy environment. CUDA 13.0 PyTorch wheels target a compatible NVIDIA driver.

For Jupyter:

```bash
uv sync --locked --extra notebook
uv run --extra notebook jupyter lab simple.ipynb
```

Select this project's Python kernel. In Colab, put the project at `/content/h3_simple`, install uv, then run `!uv --directory /content/h3_simple sync --locked` followed by `!uv --directory /content/h3_simple run main.py`. Use the subprocess commands to keep its packages separate from Colab's existing kernel.

The readable entry point is short; `h3_functions.py` contains the actual H3 loading and experimental SelfLift implementation. No ComfyUI dependency. Small real-H3 tensor/LoRA tests passed for the helper, but full pretrained standalone rendering has not been verified.

**Hardware:** approximately 75 GB host RAM for the int8 offload recipe (128 GiB recommended), native BF16 GPU, and at least 140 GiB free model-cache storage. The current T4/13 GB runtime cannot run this loader. Capacity checks stop before downloading pretrained models; installing Python packages still downloads large PyTorch wheels. Simplifying the entry point does not reduce model memory requirements.

Credits: [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3), [H3 Turbo LoRA](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI), [Diffusers](https://github.com/huggingface/diffusers), and [SelfLift research](https://arxiv.org/abs/2609.02036). These dependencies retain their own licenses. The original working Comfy notebook remains available.

LangChain is not included: the renderer does not call a language model. It can be added later for prompt writing or scene planning without changing the PyTorch generation loop.

## Video samples

Three real video samples, previews and measurements are in [samples/](samples/README.md): [ceramic studio](samples/ceramic-studio.mp4), [rainy tea](samples/rainy-tea.mp4), and [1080p portrait](samples/portrait-1080p.mp4). These came from the earlier Comfy backend and are not evidence of full-model standalone execution.

## Can it run on a 12 GB GPU?

**Potentially at a small canvas, with substantial CPU RAM and offloading; this project has not been validated on a 12 GB GPU.** The [official Diffusers H3 memory guide](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3#memory) describes 12–16 GB GPUs with int8 weights, offloaded video VAE and a small 960×544 canvas, but around 75 GB of host RAM. This project's loader requires at least 80 GiB total host RAM, native BF16 GPU support and sufficient model-cache disk space; 128 GiB host RAM is recommended. SelfLift's transition can add memory pressure beyond an ordinary Turbo-only render.

The previous native 1080p sample used around 13.6 GiB of device memory on the T4, so it is not evidence that 1080p fits 12 GB. A 12 GB GPU plus a standard 13 GB RAM Colab runtime cannot use this standalone loader as written.
