from tiny_transformer import tiny_forward


def test_shapes_and_causal_mask() -> None:
    trace = tiny_forward([0, 1, 2])
    assert len(trace["x"]) == 3 and len(trace["x"][0]) == 4
    assert len(trace["weights"]) == 3 and len(trace["weights"][0]) == 3
    assert len(trace["logits"]) == 3 and len(trace["logits"][0]) == 5
    for row_index, row in enumerate(trace["weights"]):
        assert abs(sum(row) - 1.0) < 1e-9
        assert all(weight < 1e-100 for weight in row[row_index + 1 :])


def test_is_deterministic() -> None:
    assert tiny_forward([0, 1, 2])["logits"] == tiny_forward([0, 1, 2])["logits"]


if __name__ == "__main__":
    test_shapes_and_causal_mask()
    test_is_deterministic()
    print("tiny_transformer: all checks passed")
