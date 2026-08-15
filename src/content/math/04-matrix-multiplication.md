---
title: "Matrix multiplication"
order: 4
summary: "多数の線形変換をまとめて表す計算"
concepts: ["Matrix multiplication"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Matrix multiplication

## これは何か

多数の線形変換をまとめて表す計算。`[T,d] × [d,h] → [T,h]` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

Q/K/V の projection は、各 token に同じ重み行列を掛ける行列積で書ける。

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
