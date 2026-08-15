"""A small deterministic decoder-only Transformer for teaching, not performance.

This file deliberately keeps attention visible. Tensor shapes use plain nested
Python lists: [T, d_model] for token states and [T, vocab] for logits.
See chapters 06, 08, 15, 19, 26, and 27.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

Vector = list[float]
Matrix = list[Vector]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    """Multiply [a, b] by [b, c] without a high-level attention call."""
    assert left and right and len(left[0]) == len(right)
    result = zeros(len(left), len(right[0]))
    for i in range(len(left)):
        for j in range(len(right[0])):
            result[i][j] = sum(left[i][k] * right[k][j] for k in range(len(right)))
    return result


def transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def add(left: Matrix, right: Matrix) -> Matrix:
    return [[a + b for a, b in zip(row_l, row_r)] for row_l, row_r in zip(left, right)]


def relu(matrix: Matrix) -> Matrix:
    return [[max(0.0, value) for value in row] for row in matrix]


def softmax(values: Vector) -> Vector:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def causal_attention(x: Matrix, wq: Matrix, wk: Matrix, wv: Matrix) -> tuple[Matrix, Matrix]:
    """Return context [T, d_head] and causal attention weights [T, T]."""
    q, k, v = matmul(x, wq), matmul(x, wk), matmul(x, wv)
    d_head = len(q[0])
    raw_scores = matmul(q, transpose(k))
    weights: Matrix = []
    for i, score_row in enumerate(raw_scores):
        masked = [score / math.sqrt(d_head) if j <= i else -1e9 for j, score in enumerate(score_row)]
        weights.append(softmax(masked))
    return matmul(weights, v), weights


def random_matrix(rows: int, cols: int, rng: random.Random) -> Matrix:
    return [[rng.uniform(-0.35, 0.35) for _ in range(cols)] for _ in range(rows)]


def tiny_forward(token_ids: Iterable[int]) -> dict[str, Matrix]:
    """Run a 1-head, 1-block forward pass with fixed pseudo-random weights."""
    ids = list(token_ids)
    vocab, d_model, d_head, d_hidden = 5, 4, 2, 6
    assert ids and all(0 <= token_id < vocab for token_id in ids)
    rng = random.Random(7)
    embedding = random_matrix(vocab, d_model, rng)
    position = random_matrix(len(ids), d_model, rng)
    wq, wk, wv = (random_matrix(d_model, d_head, rng) for _ in range(3))
    wo = random_matrix(d_head, d_model, rng)
    w1, w2 = random_matrix(d_model, d_hidden, rng), random_matrix(d_hidden, d_model, rng)
    w_vocab = random_matrix(d_model, vocab, rng)

    # [T] -> [T, d_model]: token lookup plus a deliberately simple position table.
    x = [[embedding[token_id][feature] + position[row][feature] for feature in range(d_model)] for row, token_id in enumerate(ids)]
    context, weights = causal_attention(x, wq, wk, wv)  # [T, d_head], [T, T]
    attention_out = matmul(context, wo)  # [T, d_model]
    residual_1 = add(x, attention_out)
    mlp_out = matmul(relu(matmul(residual_1, w1)), w2)  # [T, d_model]
    hidden = add(residual_1, mlp_out)
    logits = matmul(hidden, w_vocab)  # [T, vocab]
    return {"x": x, "weights": weights, "hidden": hidden, "logits": logits}


if __name__ == "__main__":
    trace = tiny_forward([0, 1, 2])
    for name, matrix in trace.items():
        print(f"{name:>7}: [{len(matrix)}, {len(matrix[0])}]")
