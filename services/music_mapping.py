import math
from typing import Dict, List, Optional

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi: Optional[int]) -> str:
    if midi is None:
        return ""
    octave = (midi // 12) - 1
    pc = PITCH_CLASSES[midi % 12]
    return f"{pc}{octave}"


def degree_token_for_midi(
    midi: int,
    key_tonic: str = "C",
    mode: str = "major",
    accidental_pref: str = "sharps",
) -> str:
    tonic_pc = _pitch_class_index(key_tonic)
    if tonic_pc is None:
        tonic_pc = 0
    pitch_class = midi % 12
    scale = _scale_intervals(mode)
    degrees = [(tonic_pc + interval) % 12 for interval in scale]
    if pitch_class in degrees:
        degree_idx = degrees.index(pitch_class) + 1
        return str(degree_idx)
    # accidental
    closest = min(
        degrees, key=lambda d: min((pitch_class - d) % 12, (d - pitch_class) % 12)
    )
    degree_idx = degrees.index(closest) + 1
    diff = (pitch_class - closest) % 12
    if diff == 1:
        prefix = "#"
    elif diff == 11:  # -1 mod 12
        prefix = "b"
    else:
        prefix = "#" if accidental_pref == "sharps" else "b"
    return f"{prefix}{degree_idx}"


def map_notes_to_numbers(
    notes: List[Dict], key_tonic: str, mode: str = "major", acc_pref: str = "sharps"
) -> List[str]:
    numbers: List[str] = []
    for n in notes:
        midi = n.get("midi")
        if midi is None:
            continue
        numbers.append(degree_token_for_midi(int(midi), key_tonic, mode, acc_pref))
    return numbers


def _pitch_class_index(name: str) -> Optional[int]:
    name = (name or "").strip().upper()
    aliases = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    name = aliases.get(name, name)
    try:
        return PITCH_CLASSES.index(name)
    except ValueError:
        return None


def _scale_intervals(mode: str) -> List[int]:
    mode = (mode or "major").lower()
    if mode == "minor":
        return [0, 2, 3, 5, 7, 8, 10]
    return [0, 2, 4, 5, 7, 9, 11]
