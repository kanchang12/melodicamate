from services.scoring import compare_sequences, expected_numbers_for_exercise


def test_expected_numbers():
    nums = expected_numbers_for_exercise("c_major_scale_up", "C", "major")
    assert nums[0] == "1" and nums[-1] == "1"


def test_compare_sequences_partial():
    expected = ["1", "2", "3"]
    played = ["1", "2", "4"]
    acc, wrong = compare_sequences(expected, played)
    assert acc == 66.67
    assert wrong[0]["index"] == 2
