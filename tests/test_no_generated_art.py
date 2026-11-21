from pathlib import Path


def test_production_montage_contains_no_generated_art_api_path():
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "ai_viral_montage_clips.py",
        root / "freedom_clips" / "prompts" / "montage.py",
        root / "freedom_clips" / "ffmpeg.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files).lower()
    banned = [
        "client.images.generate",
        "gpt-image-1",
        "image_prompt",
        "visual_source",
    ]
    for phrase in banned:
        assert phrase not in source
