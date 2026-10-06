"""Adapt LightX2V's native H3 QKV layout to the selected Comfy-format DiT."""
from diffsynth.utils.lora.minimax_h3 import MiniMaxH3LoRALoader, MiniMaxH3LoRAConverter


class HybridTurboLoader(MiniMaxH3LoRALoader):
    def convert_state_dict(self, state_dict, suffix=".weight"):
        if not self.is_lightx2v_format(state_dict):
            raise ValueError("HybridTurboLoader expects the pinned LightX2V adapter format")
        aligned = MiniMaxH3LoRAConverter.align_to_diffsynth_format(state_dict)
        for key, value in aligned.items():
            if key.endswith(".qkv_proj.lora_B.default.weight"):
                heads = value.shape[0] // (3 * 128)
                aligned[key] = value.reshape(heads, 3, 128, -1).permute(1, 0, 2, 3).reshape_as(value)
        return super().convert_state_dict(aligned, suffix=suffix)
