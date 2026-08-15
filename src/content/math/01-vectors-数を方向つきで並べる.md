---
title: "Vectors: 数を方向つきで並べる"
order: 1
summary: "順番のある数の列としてのベクトル"
concepts: ["Vectors: 数を方向つきで並べる"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Vectors: 数を方向つきで並べる

## これは何か

順番のある数の列としてのベクトル。`vector = [x₁, x₂, …, x_d]` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

embedding、query、key、value は、同じ長さの数の列として計算される。

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
