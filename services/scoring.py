from typing import Dict, List, Tuple

EXPECTED_EXERCISES: Dict[str, Dict] = {
    "c_major_scale_up": {
        "name": "C Major Scale (up)",
        "base_numbers": ["1", "2", "3", "4", "5", "6", "7", "1"],
        "mode": "major",
        "tonic": "C",
    },
    "c_major_scale_updown": {
        "name": "C Major Scale (up/down)",
        "base_numbers": ["1", "2", "3", "4", "5", "6", "7", "1", "7", "6", "5", "4", "3", "2", "1"],
        "mode": "major",
        "tonic": "C",
    },
    "arpeggio_1358_c": {
        "name": "Arpeggio 1-3-5-8 in C",
        "base_numbers": ["1", "3", "5", "1"],
        "mode": "major",
        "tonic": "C",
    },
    "ode_to_joy_simple": {
        "name": "Ode to Joy (preset)",
        "base_numbers": ["3", "3", "4", "5", "5", "4", "3", "2", "1", "1", "2", "3", "3", "2", "2"],
        "mode": "major",
        "tonic": "C",
    },
}


def expected_numbers_for_exercise(exercise_id: str, tonic: str, mode: str) -> List[str]:
    base = EXPECTED_EXERCISES.get(exercise_id, {}).get("base_numbers", [])
    # Already scale-degree numbers, so tonic/mode used only for naming parity.
    return base


def compare_sequences(expected: List[str], played: List[str]) -> Tuple[float, List[Dict]]:
    if not expected:
        return 0.0, []
    wrong_notes: List[Dict] = []
    matches = 0
    for idx, exp in enumerate(expected):
        got = played[idx] if idx < len(played) else None
        if got == exp:
            matches += 1
        else:
            wrong_notes.append({"expected": exp, "got": got, "index": idx})
    accuracy = (matches / len(expected)) * 100
    return round(accuracy, 2), wrong_notes


def build_mistake_summary(wrong_notes: List[Dict]) -> Dict:
    if not wrong_notes:
        return {"issues": [], "summary": "Great job—no mistakes detected."}
    top = wrong_notes[:3]
    summary = "; ".join(
        [f"at {w['index']+1}: expected {w['expected']} got {w.get('got') or 'none'}" for w in top]
    )
    return {"issues": top, "summary": summary}
