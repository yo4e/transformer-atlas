---
title: "Functions"
order: 6
summary: "入力を出力に対応させる規則"
concepts: ["Functions"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Functions

## これは何か

入力を出力に対応させる規則。`f: input → output` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

層は parameter を持つ関数であり、同じ形の入力に同じ手順を適用する。

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
