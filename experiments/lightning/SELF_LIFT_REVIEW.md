# SelfLift implementation review

Reviewed the full [SelfLift v2 paper](https://arxiv.org/pdf/2609.02036v2), including its appendices, the [author-linked implementation](https://github.com/sgl-project/sglang/pull/33733), and the pinned DiffSynth H3 backend.

The earlier sample used a binary correction mask: selected locations were completely replaced by the pixel/VAE anchor. This omitted Eq. 8's adaptive correction strength. The corrected implementation computes channel-mean absolute disagreement, selects high-risk locations independently for each sample, normalizes risk within the selected region and applies weights between 0.5 and 1.0. The default selected fraction is now 0.4. Hand-calculated regression checks distinguish this from full replacement, including equal-risk and zero-residual cases.

The existing H3 velocity sign, VAE normalization and cached-clean-estimate convention were correct. The higher-resolution packed layout is rebuilt while audio latents and their schedule continue. No extra denoiser evaluation is added. The wrapper now validates dimensions and transition placement and resets its state for a subsequent generation.

The original 320×192 → 640×384, four-step sample remains in `results/` as a legacy experiment. Its output and measurements do not establish paper-faithful quality. The new preset starts at 640×384 and lifts to 800×480 before evaluation seven of eight, using the matching eight-step Turbo adapter. Sampling grids remain those expected by H3 and that adapter.

The paper's main evaluations concern image backbones. Its preliminary Wan video discussion warns about unsuitable low-resolution token distributions; it contains no H3 validation. Our smaller resolution jump is a conservative experiment, not proof of H3's native-quality regime. This remains a training-free SelfLift-zero adaptation; a Turbo adapter does not provide the trained components required for SelfLift-rich.

The new story changes prompt, duration, precision, adapter and resolution together. A better-looking output would therefore not isolate the effect of the correction fix.
