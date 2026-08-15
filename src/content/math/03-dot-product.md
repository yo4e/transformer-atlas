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

**なぜTransformerのどの箇所で必要か（要点）**
Transformer（Vaswani et al., 2017）において「dot product（内積）」は注意機構（attention）の中心的な原料です。具体的には、クエリ（Query）ベクトル q とキー（Key）ベクトル k の対応する要素ごとの積の和 q·k が attention score（生のスコア）になります。これらのスコアを d_k の平方根で割る（scaled dot‑product）ことで値のスケールを安定化し、その後 softmax を取って重み（注意重み）を得ます。重みは確率的に解釈できるようになりますが、内積そのものは確率ではなく、生の相関（符号と大きさの寄与を含む）です。以降の説明は原著（Vaswani et al., 2017）の記述に従い、後年の変種や正則化・代替関数とは区別して考えます。

**最小の定義（直感と式）**
ベクトル x = [x1, x2, …, xn] と y = [y1, y2, …, yn] の dot product（内積）は、対応する要素の積を足し合わせたものです。式で書くと
x·y = Σ_i x_i y_i
これ以外の前提（行列微分や固有値など）は不要です。要点は「位置iの値同士を掛けて全部足す」という操作であり、それぞれの要素の符号（正負）と大きさが値に寄与します。

**記号と shape の解読（Transformer 文脈）**
- q, k は d_k 次元のベクトル（形状 d_k）。ミニバッチや系列を含めると Q は (N_q × d_k)、K は (N_k × d_k) の行列になり得ます。
- Attention の生スコア行列は S = Q K^T （形状 N_q × N_k）で各要素 S_{ij} = q_i · k_j。
- 原著はさらに S を √d_k で割る：S' = S / √d_k。これが scaled dot‑product attention の核心です。
- ここでの「内積」は行列積扱いでも、個々のベクトルの対応要素の積和として解釈できます。

**逐一計算（必須の数値例）**
以下は手順を省略せずに逐一計算します。まずベクトルを定めます。
v = [2, −1, 1]
u1 = [3, 2, −2]
u2 = [2, −1, 0]

1) v·u1 の計算
要素ごとの積：
2 × 3 = 6
(−1) × 2 = −2
1 × (−2) = −2
これらを足すと 6 + (−2) + (−2) = 2。したがって v·u1 = 2。ここでは第1成分が強い正の寄与を与え、第2・第3成分が負の寄与で打ち消して合計が小さくなっています。

2) v·u2 の計算
要素ごとの積：
2 × 2 = 4
(−1) × (−1) = 1
1 × 0 = 0
合計は 4 + 1 + 0 = 5。したがって v·u2 = 5。すべての寄与が非負で合算されるため、より大きな正の内積になっています。

この比較で見えることは、内積は単に「類似さの度合い（角度）」だけでなく、各成分の大きさと符号が直接合算されるという点です。u1 は一部で符号が異なる成分があり、そのため正の寄与を打ち消す負の寄与が生じています。

**cosine similarity（余弦類似度）との違い**
cosine similarity は内積をノルム（長さ）で正規化したものです。式は cos(x,y) = (x·y) / (||x|| ||y||)。内積は長さ（ノルム）にも比例しますが、cosine は「角度」だけを見る指標です。上の例で計算してみます（数値は小数第4位程度で示します）：

||v|| = sqrt(2^2 + (−1)^2 + 1^2) = sqrt(6) ≈ 2.4495
||u1|| = sqrt(3^2 + 2^2 + (−2)^2) = sqrt(17) ≈ 4.1231
||u2|| = sqrt(2^2 + (−1)^2 + 0^2) = sqrt(5) ≈ 2.2361

cos(v,u1) = 2 / (sqrt(6) * sqrt(17)) ≈ 0.1980
cos(v,u2) = 5 / (sqrt(6) * sqrt(5)) ≈ 0.9129

ここから分かるのは、v·u1 = 2、v·u2 = 5 と内積の大小関係は同じ方向ですが、cosine が示す「角度に基づく類似度」は、ベクトルの長さに依存しない別の尺度であることです。もし u2 をスカラーで大きくすると（例: u2' = 10·u2）内積は10倍になりますが cosine は不変（正のスカラーであれば）です。つまり、Transformer の attention では内積がそのまま使われるため、ノルムの大きさもスコアに影響します（だから √d_k でスケーリングする）。

**短い透明な Python 実装（実行可能）**
以下は実際に動く Python コードで、要素寄与と内積・ノルム・cosine を表示します。

```python
import numpy as np

v = np.array([2, -1, 1], dtype=float)
u1 = np.array([3, 2, -2], dtype=float)
u2 = np.array([2, -1, 0], dtype=float)

def explain_dot(a, b):
    products = a * b
    print("a =", a)
    print("b =", b)
    print("element-wise products:", products)
    print("dot product:", products.sum())
    print("||a||, ||b||:", np.linalg.norm(a), np.linalg.norm(b))
    print("cosine similarity:", products.sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    print()

explain_dot(v, u1)
explain_dot(v, u2)
```

このコードは要素ごとの積を明示し、どの成分がスコアを押し上げ／押し下げしているかを透明にします。

**よくある誤解（整理）**
- 「内積＝cosine similarity」と混同する：誤り。内積はノルムにも依存する。cosine は角度のみを見る。
- 「内積は確率」：誤り。内積は相関（正負と大きさ）であり、softmax を通して初めて確率的重みになる。
- 「符号は無視してよい」：誤り。負の寄与は softmax の前では重要であり、相対的に小さい確率を作る。
- 「内積は距離」：誤り。距離はノルムの差に関係し、内積は角度・長さの組合せを表す別物。

**数が何を表すか（直感）**
- 内積が大きい（正）: q と k が「同じ方向に強い成分」を持っている（かつ大きいノルム）。Transformer ではそのキーに注意を多く向ける可能性が高い。
- 小さいか負: 成分で打ち消し合っているか、逆向きの成分がある。softmax 後は低い重みになる。
- スケーリング（√d_k）: 次元数が大きいと内積の分散が増えるため、スケーリングで値を落ち着かせる。

**読者が説明できること（到達目標）**
この稿を読んだ後、あなたは以下を説明できるはずです。
- dot product を要素ごとに計算して寄与（正負・大きさ）を読み取る方法。
- Transformer の attention において q·k がどこで使われるか（QK^T、√d_k でのスケーリング、softmaxで重み化）を簡潔に説明すること。
- dot product と cosine similarity の違いを数値例で示して説明すること。
- attention スコアがどのようにして注意重みへ変換されるか（内積 → スケーリング → softmax）を説明すること。

**関連章へのリンク（本文内参照）**
- [線形代数の基礎（内積とノルム）](./linear_algebra.md)
- [Scaled dot‑product attention の詳細と実装](./scaled_dot_product_attention.md)
- [softmax と確率化の直感](./softmax_intuition.md)
- [Vaswani et al., 2017（原著）](https://arxiv.org/abs/1706.03762)

**参照**
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. arXiv:1706.03762.

第1稿。手計算の速さは要求しませんが、上の例を紙に書き出せば内積の意味が直感的に掴めるはずです。
