"""Edit the settings below, then run: uv run main.py"""
from h3_functions import (
    VideoSettings, load_models, apply_lora, configure_memory, generate_video,
)


def main():
    settings = VideoSettings(
        prompt="A woman gently tilts her head and smiles. Natural motion, steady camera, quiet room ambience.",
        image_path=None,  # Change to "photo.jpg" for image-to-video.
        width=544,
        height=960,
        seconds=5,
        seed=9174,
    )
    pipe = load_models(cache_dir="models")
    apply_lora(pipe, strength=settings.turbo_strength)
    configure_memory(pipe)
    video = generate_video(pipe, settings, output_dir="output", use_selflift=True)
    print("Your video:", video)


if __name__ == "__main__":
    main()
