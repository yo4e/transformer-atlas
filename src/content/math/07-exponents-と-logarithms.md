---
title: "Exponents と logarithms"
order: 7
summary: "softmax と loss に現れる二つの逆向きの演算"
concepts: ["Exponents と logarithms"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Exponents と logarithms

## これは何か

softmax と loss に現れる二つの逆向きの演算。`softmax(z)_i = exp(z_i) / Σ exp(z_j)` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

指数は score 差を強調し、log は正解確率が小さいときの損失を大きく表す。

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
