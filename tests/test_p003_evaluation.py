from scripts.generate_p003_evaluation import EVENTS, SEEDS, evaluation_requests


def test_evaluation_matrix_has_normal_and_paired_event_controls() -> None:
    rows = evaluation_requests("Speak clearly and naturally.")
    assert len(rows) == len(SEEDS) + len(EVENTS) * len(SEEDS) * 2
    assert len({row["id"] for row in rows}) == len(rows)
    for event, tag in EVENTS:
        for seed in SEEDS:
            tagged = next(
                row for row in rows if row["id"] == f"{event}_tagged_seed-{seed}"
            )
            control = next(
                row for row in rows if row["id"] == f"{event}_control_seed-{seed}"
            )
            assert tagged["text"] == f"{tag} {control['text']}"
            assert tagged["instruction"] == control["instruction"]
