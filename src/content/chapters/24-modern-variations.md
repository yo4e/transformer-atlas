---
title: "Modern variations"
part: 6
order: 24
summary: "core Transformer の周囲にある代表的な設計選択"
prerequisites: []
concepts: ["Modern variations"]
interactive_components: []
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
status: complete
---

# Modern variations

## 問題

core Transformer の周囲にある代表的な設計選択。この章では、全体のどこでこの部品が必要になり、何を受け取り、何を返すかを先に固定する。

## 直感

近年の設計は目的も採用状況も異なる。ここでは一つの『現代標準』として混ぜず、各手法の問題設定を分ける。

## 実際の機構

ここで扱う計算を短く書くと、`RoPE / RMSNorm / GQA / MoE` となる。記号は小さな例で確認してから使う。教育用の図と数値は概念を追うために小さくしており、実モデルの内部状態をそのまま可視化したものではない。

## 最小の数式と shape

ベクトルは数を順に並べたもの、行列はベクトルを行または列に並べたものである。shape は各軸の長さを表す。たとえば token 数を `T`、隠れ次元を `d` と書けば、入力表現 `X` は通常 `[T, d]` と書ける。

## 小さな例

三つの token を使うなら `T=3` である。各 token が四つの数を持つ場合は `X` の shape は `[3, 4]` になる。大きな値を暗算する必要はない。まず「どの軸が token で、どの軸が特徴か」を声に出して確認する。

## コードまたは擬似コード

```python
# 入力の形を明示し、次の演算の出力を検査する
assert len(x) == token_count
next_value = transform(x)
print("input", shape(x), "output", shape(next_value))
```

## この比喩の限界

直感は計算の入口であり、学習済み parameter の意味を人間の言葉で固定するものではない。特に attention の重みや embedding の一軸を、単独でモデルの思考や概念と同一視しない。

## 実際の LLM との接続

大規模な言語モデルでは同種の計算を、より多い層・head・次元・語彙で繰り返す。構成の詳細はモデルごとに異なるため、この章の例から全モデルの設計を断定しない。

## 理解を確かめる

1. 「Modern variations」で、入力と出力はそれぞれ何か。
2. この計算がなければ、どの問題が残るか。
3. 教材の小さな例と実際の大規模モデルで、同じ点と異なる点は何か。

## 一文で言うと

**近年の設計は目的も採用状況も異なる。ここでは一つの『現代標準』として混ぜず、各手法の問題設定を分ける。**

## 参照

- [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
