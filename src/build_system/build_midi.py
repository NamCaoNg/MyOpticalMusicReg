"""
XML to MIDI Converter
Đọc file MusicXML và chuyển đổi sang file MIDI.
Usage:
    python xml_to_midi.py input.xml -o output.mid
    python xml_to_midi.py input.xml --tempo 120 --instrument piano"""

import os
import sys
import argparse
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Optional
from music21 import converter, tempo as m21_tempo, instrument as m21_instrument

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Instrument mapping (General MIDI program numbers)
INSTRUMENT_MAP = {
    'piano': 0,
    'acoustic_piano': 0,
    'electric_piano': 4,
    'harpsichord': 6,
    'organ': 19,
    'guitar': 24,
    'acoustic_guitar': 25,
    'electric_guitar': 27,
    'violin': 40,
    'viola': 41,
    'cello': 42,
    'strings': 48,
    'trumpet': 56,
    'trombone': 57,
    'flute': 73,
    'clarinet': 71,
    'saxophone': 65,
}


def xml_to_midi(
    xml_path: str,
    output_path: Optional[str] = None,
    tempo: Optional[int] = None,
    instrument: Optional[str] = None) -> str:
    """
    Chuyển đổi file MusicXML sang MIDI.
    Args:
        xml_path: Đường dẫn file MusicXML đầu vào
        output_path: Đường dẫn file MIDI đầu ra (mặc định: cùng tên với .mid)
        tempo: Tempo (BPM), None để giữ nguyên từ XML
        instrument: Tên instrument (piano, guitar, violin, ...)
    Returns:
        Đường dẫn file MIDI đã tạo"""

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Không tìm thấy file: {xml_path}")

    logger.info(f"Reading XML: {xml_path}")
    score = converter.parse(xml_path)

    # Set tempo if specified
    if tempo is not None:
        logger.info(f"Đặt tempo: {tempo} BPM")
        # Remove existing tempo marks
        for element in score.recurse().getElementsByClass(m21_tempo.MetronomeMark):
            score.remove(element, recurse=True)
        # Add new tempo at the beginning
        mm = m21_tempo.MetronomeMark(number=tempo)
        score.insert(0, mm)

    # Set instrument if specified
    if instrument is not None:
        instrument_lower = instrument.lower().replace(' ', '_')
        if instrument_lower not in INSTRUMENT_MAP:
            available = ', '.join(INSTRUMENT_MAP.keys())
            raise ValueError(
                f"Instrument '{instrument}' không được hỗ trợ. "
                f"Các instrument có sẵn: {available}"
            )

        logger.info(f"Đặt instrument: {instrument}")
        program = INSTRUMENT_MAP[instrument_lower]

        for part in score.parts:
            # Remove existing instruments
            for inst in part.recurse().getElementsByClass(m21_instrument.Instrument):
                part.remove(inst, recurse=True)
            # Add new instrument
            new_inst = m21_instrument.Instrument()
            new_inst.midiProgram = program
            part.insert(0, new_inst)

    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(xml_path)[0]
        output_path = base_name + '.mid'
    elif os.path.isdir(output_path):
        base_name = os.path.splitext(os.path.basename(xml_path))[0]
        output_path = os.path.join(output_path, base_name + '.mid')
    elif not output_path.endswith(('.mid', '.midi')):
        output_path = output_path + '.mid'

    # Create output directory if needed
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Generating MIDI: {output_path}")
    score.write('midi', fp=output_path)

    logger.info(f"COMPLETE!- {output_path}")
    return output_path


def get_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(description="Chuyển đổi MusicXML sang MIDI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("xml_path", help="Input XML file path", type=str)
    parser.add_argument("-o", "--output", help="Output MIDI file path or output folder", type=str, default=None)
    parser.add_argument("-t", "--tempo", help="Tempo (BPM). Mặc định giữ nguyên từ XML", type=int, default=None)
    parser.add_argument("-i", "--instrument", help=f"Các lựa chọn: {', '.join(INSTRUMENT_MAP.keys())}", type=str, default=None)
    parser.add_argument("--list-instruments", help="List of avaiable instruments", action="store_true")
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if args.list_instruments:
        logger.info("Các instruments có sẵn:")
        for name, program in sorted(INSTRUMENT_MAP.items()):
            logger.info(f"  {name}: MIDI program {program}")
        return

    xml_to_midi(xml_path=args.xml_path, output_path=args.output, tempo=args.tempo, instrument=args.instrument)


if __name__ == "__main__":
    main()
