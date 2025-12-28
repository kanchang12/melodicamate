import difflib
from typing import Dict, List, Optional

# Simple public domain library encoded as number sequences.
PD_SONGS: List[Dict] = [
    {
        "meta": {"title": "Twinkle Twinkle Little Star", "composer": "Traditional"},
        "numbers": ["1", "1", "5", "5", "6", "6", "5", "4", "4", "3", "3", "2", "2", "1"],
    },
    {
        "meta": {"title": "Ode to Joy", "composer": "Beethoven"},
        "numbers": ["3", "3", "4", "5", "5", "4", "3", "2", "1", "1", "2", "3", "3", "2", "2"],
    },
    {
        "meta": {"title": "Mary Had a Little Lamb", "composer": "Traditional"},
        "numbers": ["3", "2", "1", "2", "3", "3", "3", "2", "2", "2", "3", "5", "5"],
    },
    {
        "meta": {"title": "C Major Scale", "composer": "Exercise"},
        "numbers": ["1", "2", "3", "4", "5", "6", "7", "1"],
    },
]


def find_song(query: str, classification: Dict) -> Optional[Dict]:
    titles = [s["meta"]["title"] for s in PD_SONGS]
    matches = difflib.get_close_matches(query, titles, n=1, cutoff=0.5)
    if not matches:
        return None
    title = matches[0]
    for song in PD_SONGS:
        if song["meta"]["title"] == title:
            return song
    return None


PITCH_CLASS_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def number_to_note_name(token: str, tonic: str = "C", mode: str = "major") -> str:
    tonic_idx = PITCH_CLASS_ORDER.index(tonic.upper())
    scale = [0, 2, 4, 5, 7, 9, 11] if mode == "major" else [0, 2, 3, 5, 7, 8, 10]
    accidental = 0
    if token.startswith("#"):
        accidental = 1
        token = token[1:]
    elif token.startswith("b"):
        accidental = -1
        token = token[1:]
    try:
        degree = int(token)
    except ValueError:
        degree = 1
    pc = (tonic_idx + scale[degree - 1] + accidental) % 12
    octave = 4
    return f"{PITCH_CLASS_ORDER[pc]}{octave}"
