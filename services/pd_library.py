import difflib
from typing import Dict, List, Optional

# Simple public domain library encoded as number sequences.
PD_SONGS: List[Dict] = [
    {
        "meta": {"title": "Twinkle Twinkle Little Star", "composer": "Traditional"},
        "numbers": ["1", "1", "5", "5", "6", "6", "5", "4", "4", "3", "3", "2", "2", "1"],
        "lyrics": "Twinkle, twinkle, little star, how I wonder what you are",
    },
    {
        "meta": {"title": "Ode to Joy", "composer": "Beethoven"},
        "numbers": ["3", "3", "4", "5", "5", "4", "3", "2", "1", "1", "2", "3", "3", "2", "2"],
        "lyrics": "Joyful joyful we adore thee",
    },
    {
        "meta": {"title": "Mary Had a Little Lamb", "composer": "Traditional"},
        "numbers": ["3", "2", "1", "2", "3", "3", "3", "2", "2", "2", "3", "5", "5"],
        "lyrics": "Mary had a little lamb, little lamb, little lamb",
    },
    {
        "meta": {"title": "C Major Scale", "composer": "Exercise"},
        "numbers": ["1", "2", "3", "4", "5", "6", "7", "1"],
        "lyrics": "",
    },
    {
        "meta": {"title": "Jingle Bells", "composer": "James Pierpont"},
        "numbers": ["3", "3", "3", "3", "3", "3", "3", "5", "1", "2", "3", "4", "4", "4", "4", "4", "3", "3", "3", "3", "2", "2", "3", "2", "5"],
        "lyrics": "Jingle bells, jingle bells, jingle all the way",
    },
    {
        "meta": {"title": "Amazing Grace", "composer": "Traditional"},
        "numbers": ["1", "3", "1", "4", "1", "5", "5", "6", "5", "4", "3", "1", "3", "1"],
        "lyrics": "Amazing grace how sweet the sound",
    },
    {
        "meta": {"title": "Greensleeves", "composer": "Traditional"},
        "numbers": ["5", "6", "5", "4", "3", "4", "5", "2", "3", "4", "1", "2", "3"],
        "lyrics": "Alas, my love, you do me wrong",
    },
    {
        "meta": {"title": "Auld Lang Syne", "composer": "Traditional"},
        "numbers": ["5", "5", "6", "5", "4", "2", "5", "5", "6", "5", "1", "7"],
        "lyrics": "Should auld acquaintance be forgot",
    },
    {
        "meta": {"title": "London Bridge", "composer": "Traditional"},
        "numbers": ["5", "6", "5", "4", "3", "4", "5", "1", "3", "5", "4", "3", "4", "5", "1"],
        "lyrics": "London Bridge is falling down",
    },
    {
        "meta": {"title": "Yankee Doodle", "composer": "Traditional"},
        "numbers": ["1", "1", "2", "3", "1", "3", "2", "1", "1", "1", "2", "3", "1", "7", "1"],
        "lyrics": "Yankee Doodle went to town",
    },
    {
        "meta": {"title": "When the Saints Go Marching In", "composer": "Traditional"},
        "numbers": ["1", "3", "4", "5", "1", "3", "4", "5", "5", "6", "5", "4", "3", "1", "2", "3", "4", "2", "1"],
        "lyrics": "Oh when the saints go marching in",
    },
    {
        "meta": {"title": "Camptown Races", "composer": "Stephen Foster"},
        "numbers": ["1", "2", "3", "1", "1", "2", "3", "1", "3", "4", "3", "2", "1"],
        "lyrics": "Camptown ladies sing this song, doo-dah, doo-dah",
    },
]


def find_song(query: str, classification: Dict) -> Optional[Dict]:
    titles = [s["meta"]["title"] for s in PD_SONGS]
    matches = difflib.get_close_matches(query, titles, n=1, cutoff=0.45)
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
