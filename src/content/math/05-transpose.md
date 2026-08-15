---
title: "Transpose"
order: 5
summary: "行と列を入れ替える操作"
concepts: ["Transpose"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Transpose

## これは何か

行と列を入れ替える操作。`(QKᵀ)[i,j] = Q[i]·K[j]` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

key の行を列に替えると、全 query と全 key の組を一つの score matrix として計算できる。

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
