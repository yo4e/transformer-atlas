---
title: "Coordinates と dimensions"
order: 2
summary: "座標と次元が何を数えているか"
concepts: ["Coordinates と dimensions"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Coordinates と dimensions

## これは何か

座標と次元が何を数えているか。`shape([T, d]) = token axis × feature axis` と書くと、Transformer の計算で現れる関係を短く表せる。

## なぜここで必要か

次元は人が個別に名付ける属性の一覧ではなく、モデルが学ぶ表現空間の座標数である。

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
