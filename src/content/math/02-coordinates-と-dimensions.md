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

Transformerを実装・理解するとき、数値が「何を数えているか」を正確に把握することが非常に重要です。特に axis（軸）、coordinate（座標）、dimension（次元）、shape（形状）の違いをあいまいにすると、バグや概念の誤解が生じます。本稿では大学数学の前提なしに、特に「token axis（系列＝時刻軸）」と「feature axis（特徴＝埋め込み次元）」を混同しないための最小限の定義と実例を示します。最後に短い実行可能なPythonコードで確認できます。原著のTransformerに関する事実は Vaswani et al., 2017 に従い、実装上の軸順（例：batch-first か time-major か）は実装ごとに異なることを明記します。

**なぜTransformerのどの箇所で必要か**
- 入力表現（単語やトークンの埋め込み）は「各トークン（位置）に対して d_model 個の値」を持ちます。ここで「トークンの数」を数えるのが token axis（通常 T）、各トークンが持つ値の数を数えるのが feature axis（通常 d）。
- Self-attentionは「トークン同士の相互作用」を計算するため、トークン軸に沿った類似度計算（例えば内積）を行います。feature axis はその内積の材料（座標）です。
- 線形変換（W x）やLayerNorm、バッチ処理、マスク、ブロードキャストなどはいずれも「どの軸に沿って計算するか」を前提にコードを書かなければならず、axisの取り違えで結果が全く異なります。Vaswani et al.（2017）はトークンごとに d_model 次元ベクトルを扱う点を設計上の前提にしていますが、実装のメモリレイアウト（例：[B, T, d] vs [T, B, d]）は環境依存です。

**最小の定義**
- axis（軸）: 配列の「位置の向き」。例：axis=0 が行（token axis）、axis=1 が列（feature axis）と解釈することが多い。
- coordinate（座標）: ある軸に沿った個々の値。行列のある位置にある単一の数。
- dimension（次元）: ある軸に沿った要素の数（length）。T や d の値そのもの。
- shape（形状）: 全軸の次元を並べたもの。例えば [T, d] は T 個のトークン、各トークンが d 次元ベクトルであることを示す。

以下、具体的な数値例で「何が変わるか」を手で追って説明します。

**数値例の完全計算（省略なし）**
与えられた行列（3 トークン × 2 フィーチャ）:
| token \ feature | f1 | f2 |
|---:|---:|---:|
| t1 | 1 | 0 |
| t2 | 2 | 1 |
| t3 | -1 | 3 |

これを行列で書くと A = [[1,0],[2,1],[-1,3]]。形状は [T=3, d=2] です。意味づけは「3つのトークン、それぞれが2次元ベクトルを持つ」。

トークンベクトル（行ごと）:
- t1 = [1, 0]
- t2 = [2, 1]
- t3 = [-1, 3]

フィーチャ列（列ごと）:
- f1 column = [1, 2, -1]（この列は「各トークンの第1座標」を並べたもの）
- f2 column = [0, 1, 3]（「各トークンの第2座標」）

トランスポーズ（行列転置）A^T = [[1,2,-1],[0,1,3]]。形状は [2,3]。ここでの意味は「2行（元の特徴数）×3列（元のトークン数）」で、行を特徴、列をトークンと解釈するならば軸が入れ替わったことを示します。数値自体は入れ替えただけで不変ですが、同じ計算（例：トークン間類似度）を行う前提が崩れます。

具体的な計算例：トークン間の内積（類似度）
- t1·t2 = 1*2 + 0*1 = 2
- t1·t3 = 1*(-1) + 0*3 = -1
- t2·t3 = 2*(-1) + 1*3 = 1

もしデータを誤って A^T のまま「各行がトークン」だと解釈して同じ内積式を適用すると、計算は次のように変わる（A^T の行を新しい「トークン」として計算）:
- row1(A^T)·row2(A^T) = [1,2,-1]·[0,1,3] = 1*0 + 2*1 + (-1)*3 = -1

これは元の t1·t2（=2）と全く異なる値になり、意味が失われます。したがって「行列のshapeが入れ替わると、同じ数値でも『何を数えているか』が変わる」ことを必ず意識してください。

**記号と shape の解読**
- [T, d] : 通常は T（時系列長、トークン数）を第一軸、d（埋め込み次元）を第二軸とする表現。
- [B, T, d] : バッチを含めた形。B はバッチサイズ。Vaswani et al. の紙は各トークンを d_model 次元で表す点を前提としていますが、図の中での軸の順序は説明のための抽象化であり、実装の細部（TensorFlow や PyTorch の API）は各ライブラリの慣習に従います。
- Multi-head attention の内部では、しばしば形状が [B, T, n_heads, d_k] や [B, n_heads, T, d_k] のように変形・転置されます。ここで重要なのは目的（トークンごとに head ごとの特徴を計算する）であり、整数値の位置を間違えないことです。

**透明な短い Python（実行可能）**
以下は NumPy を使った「行列の形」と「トークン内積」の確認コードです（実行可能）。
```python
import numpy as np

A = np.array([[1,0],
              [2,1],
              [-1,3]])   # shape (3,2) -> (T=3, d=2)

sim_tt = A @ A.T      # shape (3,3), sim_tt[i,j] = ti · tj
print("sim_tt:\n", sim_tt)

# 列（特徴）を表示
print("feature columns:\n", A[:,0], A[:,1])

# 転置してから同じやり方で計算（誤った解釈の例）
AT = A.T              # shape (2,3)
sim_AT = AT @ AT.T    # shape (2,2)
print("sim_AT (wrong interpretation):\n", sim_AT)
```
実行すると sim_tt の値は先述の内積を含み、sim_AT はまったく別の意味の行列になります。

**よくある誤解**
- 「行列の転置は単に見た目を変えるだけ」：数値は同じでも「行が何を表すか」「列が何を表すか」が変われば計算結果の意味は別物になります。
- 「axis の番号はどのライブラリでも同じ意味」：多くは axis=0 が最外軸（行）ですが、フレームワークや関数（例：RNN の time-major オプション）によって期待される軸順が異なります。
- 「broadcast で自動的に合わせてくれるから axis を気にしなくてよい」：ブロードキャストは数値的に動くが、意味（トークン対トークンの計算か、特徴ごとの操作か）を保証しません。
- Attention の形状を扱うときに [B, T, H, D] と [B, H, T, D] を混同すること。転置が必要な箇所を見落とすとスコアが壊れる。

**本文の関連章へのリンク**
- Vaswani et al., "Attention Is All You Need"（原著）: https://arxiv.org/abs/1706.03762
- Self-Attention（概念と実装）: https://arxiv.org/abs/1706.03762 （論文中の self-attention と multi-head の節を参照）
- NumPy transpose/docs: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
- PyTorch tensor shape conventions: https://pytorch.org/docs/stable/tensors.html

**読者が説明できること（到達目標）**
- [ ] token axis と feature axis の違いを言語化できる。
- [ ] 行列 A=[T,d] と A^T=[d,T] が「同じ数値でも何を表すか」がどう変わるかを示せる。
- [ ] attention の入力（トークンごとの d 次元ベクトル）とその形状表記を説明できる。
- [ ] 実装で axis を間違えるとどのような誤りが生じるか具体例を挙げられる。
- [ ] 短い Python スクリプトでトークン間類似度を計算し、転置と比較できる。

**参照**
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. arXiv:1706.03762.
- NumPy documentation (transpose, broadcasting).
- PyTorch documentation (tensor shapes, permute).

以上が「coordinates と dimensions」に関する第二稿的説明です。手計算の速さや高度な線形代数は不要です。重要なのは「数が何を表すか」を常に問い、コードと数学的記述の軸順が一致していることを確かめる習慣です。
