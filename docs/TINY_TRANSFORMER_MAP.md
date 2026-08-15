# Tiny Transformer と章の対応

`examples/tiny_transformer.py` は性能実験用ではなく、**式を実際の数値配列の処理へ戻すための伴走実装**である。高水準の attention API を使わず、`matmul`、`transpose`、`softmax`、mask、残差、MLP、logits を可視の関数として置いている。

| 実装箇所 | 何をしているか | 読む章 |
| --- | --- | --- |
| `embedding[token_id] + position[row]` | token ID の lookup と、位置情報の加算 | 2, 3, 10 |
| `matmul(x, wq/wk/wv)` | 同じ入力から Q/K/V の学習済み射影を計算する | 4, 6 |
| `matmul(q, transpose(k))` | 各 query と各 key の全組の score を作る | 4, 5, 8 |
| `j <= i else -1e9` | future position を softmax 前に除外する | 15 |
| `softmax(masked)` と `matmul(weights, v)` | 比較 score を混合比へ変え、value を集める | 7, 8 |
| `add(x, attention_out)` と `add(residual_1, mlp_out)` | 残差経路で入力を保ちながら block を積む | 11–13 |
| `matmul(hidden, w_vocab)` | hidden representation から語彙 logits を作る | 16 |

この実装は一つの head、一つの block、固定された擬似乱数の小さな parameter を用いる。実運用のLLMは多層・多head・大語彙・高度な数値計算・実装最適化を伴うため、このコードの出力を実モデルの挙動の縮小版とみなしてはいけない。
