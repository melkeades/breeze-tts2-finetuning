from scripts.generate_compile_quality_matrix import evaluation_requests


def test_compile_quality_matrix_is_balanced_and_paired() -> None:
    rows = evaluation_requests()
    assert len(rows) == 80
    assert {group: sum(row["group"] == group for row in rows) for group in {
        "neutral", "styled", "sound_control", "sound_tagged"
    }} == {
        "neutral": 20,
        "styled": 20,
        "sound_control": 20,
        "sound_tagged": 20,
    }

    pairs: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    assert len(pairs) == 40
    assert all(len(pair) == 2 for pair in pairs.values())
    for pair in pairs.values():
        assert pair[0]["seed"] == pair[1]["seed"]
