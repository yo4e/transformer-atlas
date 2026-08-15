# 教育用 Tiny Transformer

`examples/tiny_transformer.py` は、CPU だけで実行できる小さな decoder-only Transformer の **前向き計算** である。性能・学習品質・実運用を目指す実装ではなく、attention の行列積、causal mask、residual、MLP、logits を一つずつ確認することを目的にする。

| 中間値 | shape | 説明 | 対応章 |
| --- | --- | --- | --- |
| `x` | `[T, d_model]` | token embedding と位置表現を足した入力 | 3, 10, 27 |
| `weights` | `[T, T]` | 未来を隠した attention weight | 6–8, 15 |
| `hidden` | `[T, d_model]` | attention と MLP の residual 後の表現 | 11–13 |
| `logits` | `[T, vocab]` | 各位置の次 token 用、未正規化 score | 16, 19 |

```bash
python3 examples/tiny_transformer.py
python3 examples/test_tiny_transformer.py
```

> 本実装は LayerNorm、multi-head、dropout、学習 loop、KV cache、GPU 最適化を省略している。これらを省く理由は、教材の最初の forward trace で attention の核を隠さないためである。

