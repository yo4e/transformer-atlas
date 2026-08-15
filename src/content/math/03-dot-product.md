---
title: "Dot product"
order: 3
summary: "対応要素の積を足す比較"
concepts: ["Dot product"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Dot product

## これは何か

対応要素の積を足す比較。`a·b = Σ a_i b_i` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

attention score は query と key の dot product から始まる。

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
