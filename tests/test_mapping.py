from services.music_mapping import degree_token_for_midi, midi_to_note_name


def test_note_name():
    assert midi_to_note_name(60) == "C4"
    assert midi_to_note_name(61) == "C#4"


def test_degree_tokens_major():
    assert degree_token_for_midi(60, "C", "major") == "1"
    assert degree_token_for_midi(62, "C", "major") == "2"
    assert degree_token_for_midi(61, "C", "major") == "#1"


def test_degree_tokens_minor():
    assert degree_token_for_midi(60, "A", "minor") == "1"
    assert degree_token_for_midi(61, "A", "minor").startswith("#")
