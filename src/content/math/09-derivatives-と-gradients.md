---
title: "Derivatives と gradients"
order: 9
summary: "微小な変化への反応を測る"
concepts: ["Derivatives と gradients"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Derivatives と gradients

## これは何か

微小な変化への反応を測る。`gradient = ∂loss / ∂parameter` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

学習では parameter の各要素を少し動かしたとき、loss がどう変わるかをまとめて扱う。

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
