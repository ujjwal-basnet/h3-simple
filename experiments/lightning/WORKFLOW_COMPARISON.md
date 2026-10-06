# Independent notebook direction

This comparison describes the workflow JSON supplied by the user. It does not establish the quality, licensing, or availability of its named weights.

| Component | Supplied graph | Current standalone experiment |
| --- | --- | --- |
| H3 checkpoint | Hybrid FL2VA/REF2VA, INT8 | FL2VA, NF4 |
| Turbo adapter | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui`, strength 0.7 | LightX2V eight-step FL2V adapter |
| Additional adapters | jiandou 0.65, H3_Combat_V2 0.5, Motion_Repair 0.25 | None |
| Schedule | Eight-step simple schedule plus one refinement step | Eight-step shifted flow schedule |
| Resolution transition | Low-resolution scale 0.5, transition setting 6, named learned 3D latent upscaler | 640×384 to 800×480, interpolation and pixel-anchor correction |
| Correction | rho 0: pixel-anchor correction disabled | rho 0.4, adaptive weights 0.5–1 |
| Attention | TST followed by SolAttnMiniMax in the active links | PyTorch SDPA with guarded FP16 attention |

The graph's node titles are not authoritative: Motion_Repair is connected and has mode 0 despite its bypass title. Notes say “Sage only,” but the links contain SolAttnMiniMax. The duration input is 10 seconds, while the prompt describes 15 seconds; the frame-count expression produces 243 frames, about 10.125 seconds at 24 FPS. Resolution comes from the linked 2-megapixel selector, not the conditioning node's stored 1344×768 widgets.

The independent implementation should keep Python functions and a small notebook, use an original story and assets, and record each upstream model and method. It should first demonstrate a clean single-resolution baseline, investigate the baked-in grid artifacts, then compare the resolution transition against that same baseline. Do not call the current outputs quality-validated.

A learned H3 latent upscaler is a separate component from the interpolation-based transition currently implemented. Its filename alone does not establish compatibility or make this graph a validated implementation of SelfLift-rich. Adapter and upscaler ports require verified source, architecture, and weight mapping before use.

The notebook's original code, orchestration, prompts, and assets can be original contributions. H3, Turbo adapters, other authors' adapters, and the SelfLift research method retain their respective authorship; renaming a graph does not transfer it.
