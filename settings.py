"""One settings class; Pydantic handles types, ranges and photo paths."""
import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator


class VideoSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    prompt: str = Field(min_length=1)
    image_path: FilePath | None = None
    width: int = Field(default=544, ge=64, multiple_of=32)
    height: int = Field(default=960, ge=64, multiple_of=32)
    seconds: float = Field(default=5, ge=5, le=15)
    seed: int = Field(default=9174, ge=0, le=2**63-1)
    steps: int = Field(default=8, ge=2)
    transition_step: int = Field(default=6, ge=1)
    lowres_scale: float = Field(default=0.5, gt=0, lt=1)
    rho: float = Field(default=0.4, gt=0, le=1)
    turbo_strength: float = Field(default=1, ge=0, le=2)
    export_width: int | None = Field(default=None, gt=0, multiple_of=2)
    export_height: int | None = Field(default=None, gt=0, multiple_of=2)

    @property
    def frames(self) -> int:
        frames = math.ceil(self.seconds * 24)
        return frames + (5 - frames % 17) % 17

    @property
    def low_size(self) -> tuple[int, int]:
        return tuple(max(32, round(size*self.lowres_scale/32)*32)
                     for size in (self.height, self.width))

    @model_validator(mode="after")
    def check_related_settings(self) -> Self:
        if self.transition_step >= self.steps:
            raise ValueError("Leave at least one step after the SelfLift transition.")
        if self.frames / 24 > 15:
            raise ValueError("The aligned frame count exceeds the backend's 15-second limit.")
        if any(low >= target for low, target in zip(self.low_size, (self.height, self.width))):
            raise ValueError("The low-resolution canvas must be smaller than the target.")
        for crop, size in ((self.export_width, self.width), (self.export_height, self.height)):
            if crop is not None and crop > size:
                raise ValueError("Export crops must fit inside the generated canvas.")
        return self
