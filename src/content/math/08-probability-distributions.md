---
title: "Probability distributions"
order: 8
summary: "候補全体に割り振られた非負の重み"
concepts: ["Probability distributions"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Probability distributions

## これは何か

候補全体に割り振られた非負の重み。`p_i ≥ 0, Σ p_i = 1` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

softmax 後の次 token 分布は、指定した語彙内での相対的な選択重みである。

## 最小の計算

数字を小さくして、各記号が何を数えるかを先に確認する。演算の結果よりも、入力と出力の shape がどう変わるかを追う。

## コードとの接続

```python
# 教材用: shape を先に表示してから演算する
print(shape(input_value))
result = operation(input_value)
print(shape(result))
```

## 参照

- [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
