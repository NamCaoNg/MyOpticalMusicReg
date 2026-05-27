import os


def process_score_image(
    input_path: str,
    output_dir: str,
    *,
    without_deskew: bool = False,
    tempo: int | None = None,
    instrument: str | None = None) -> dict:
    """Run only the OMR pipeline and return generated output paths."""
    from src.main import run_pipeline

    os.makedirs(output_dir, exist_ok=True)

    result = run_pipeline(
        img_path=input_path,
        output_dir=output_dir,
        without_deskew=without_deskew,
        tempo=tempo,
        instrument=instrument,
    )

    xml_path = result.get("xml_path")
    midi_path = result.get("midi_path")

    if not xml_path or not os.path.exists(xml_path):
        raise FileNotFoundError(f"Pipeline completed but did not create XML file: {xml_path}")

    if not midi_path or not os.path.exists(midi_path):
        raise FileNotFoundError(f"Pipeline completed but did not create MIDI file: {midi_path}")

    return {
        "output_dir": result.get("output_folder", output_dir),
        "xml_path": xml_path,
        "midi_path": midi_path,
    }
