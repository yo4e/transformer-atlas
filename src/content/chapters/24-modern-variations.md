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

本章は「なぜ一部のTransformer実装が元の形からずれているのか」を、具体的な数値経路を示しながら説明します。まずは各工夫が解こうとする実際の問題を名前を出さずに提示します。

問題の提示（名前を伏せた説明）
- 長い文脈を扱うとき、位置情報を単純に加えると学習した振る舞いが遠い位置で崩れる。絶対位置をそのまま足すだけでは「距離としての情報」がうまく保存されない場合がある。
- レイヤ正規化で平均を引く操作は効果的だが、実装コストや安定性の面で代替を望むことがある。特に大規模バッチや小さいバッチで分散の扱いが変わる。
- フィードフォワード層の単純な活性化関数（ReLUやGELU）では、同じ重み数でより豊かな非線形性をほしい場面がある。
- AttentionでQueryとKey/Valueの頭数を固定で同じにしておくと、計算コストと表現力のトレードオフが硬直する。キー／値を少なくしてもよい場面がある。
- Attentionの全体行列（長さ×長さ）を一度に作るとメモリと速度が問題になりがちで、計算を分割・融合して高速化したい。
- モデルの容量を増やすために、全結合層を単純に大きくすると計算負担が増す。活かせるパラメータを選択的に使う方法が欲しい。

これらの問題に対する「後年の変種」は、それぞれ違う目的で生まれ、全てのモデルに必須というわけではありません。以降で各技術の直感・数式・短い実装例・全体の流れ内での位置付けを示します。注意：以下の小さな数値例は教科書用の簡略化であり、実際の大規模モデルの内部値ではありません。また「token」は語と同義ではなく、attention weight を人間の「注目」を示す完全な証拠と扱うのは誤りです。

RoPE（回転埋め込み）の直感と数式
直感：各位置のベクトルに、「その位置だけが持つ回転」をかけることで、相対的なずれが内積として直感的に反映されるようにする。回転は線形で、長い距離の外挿が比較的安定に働く点が利点。ただし「回転＝意味の回転」と文字通り受け取ると誤解する。回転は位相を付与する簡単な線形変換で、各次元対で行う。

数式（定義と記号）
- 入力系列のある位置 p に対する query ベクトル q(p) ∈ R^d を考える。
- d は偶数とし、次元を2つずつ（i,i+1）で扱う。
- 回転は角周波数 ω_k を用いて 2×2 回転行列 R(ω_k p) を対ごとに適用する：
  q_rot(p)_{2k:2k+1} = R(ω_k p) · q_{2k:2k+1},
  R(θ) = [[cos θ, -sin θ],[sin θ, cos θ]].
- 同様にキーにも同じ回転を適用する。内積 q_rot(p) · k_rot(t) は位置差 p−t に依存する位相差を生む。

簡単な数値例（d=4）
- q = [q0,q1,q2,q3] を 2対に分けて角周波数 ω_0=1.0, ω_1=0.5、位置 p=2 と t=0 とする。
- 対ごとに回転して内積を計算すると、位置差に応じたスケールと符号の変化が現れる（ここでは具体値をコードで示す）。

短いPython実装
```python
import math
import numpy as np

def rope_rotate(vec, pos, omegas):
    # vec: (d,), d even. omegas: list of length d//2
    v = vec.copy()
    out = np.zeros_like(v)
    for k, omega in enumerate(omegas):
        i, j = 2*k, 2*k+1
        theta = omega * pos
        c, s = math.cos(theta), math.sin(theta)
        out[i] = c * v[i] - s * v[j]
        out[j] = s * v[i] + c * v[j]
    return out

q = np.array([1.0, 0.0, 0.5, 0.5])
k = np.array([0.5, 0.5, 1.0, 0.0])
omegas = [1.0, 0.5]
q2 = rope_rotate(q, pos=2, omegas=omegas)
k2 = rope_rotate(k, pos=0, omegas=omegas)
print("q·k before", q.dot(k), "after", q2.dot(k2))
```

RMSNorm（平均を引かない正規化）の直感と数式
直感：レイヤ単位で全成分の「大きさ」(RMS: Root Mean Square) を揃える。平均を引かず、分散の平方根だけで割るため、実装が軽く、特に分散の推定に起因する数値不安定性や分散計算の通信コストを下げたい場合に有効。欠点は平均シフトを補正しない点で、モデルや学習率との相互作用を注意する必要がある。

数式（定義）
- 活性化ベクトル x ∈ R^d（入力）に対して
  rms(x) = sqrt( (1/d) Σ_i x_i^2 + ε ).
- 出力 y = (x / rms(x)) ⊙ g, ここで g ∈ R^d は学習されるスケーリング（学習パラメータ）。
- ε は小さい定数（数値安定化）。

Python実装
```python
import numpy as np

def rmsnorm(x, g, eps=1e-6):
    # x: (d,), g: (d,)
    rms = np.sqrt(np.mean(x*x) + eps)
    return (x / rms) * g

x = np.array([0.5, -1.0, 0.2, 0.0])
g = np.ones_like(x) * 0.9
print(rmsnorm(x, g))
```

SwiGLU（改良型FFN）の直感と数式
直感：フィードフォワード層を2つの線形変換に分割し、片方に滑らかなゲーティング関数（Swish = x * sigmoid(x)）をかけてそれをもう一方と掛け合わせる。掛け合わせることで非線形性が増え、同じパラメータ数で表現力が高まることが報告されている。注意：これは万能ではなく、学習率や初期化に敏感な場合がある。

数式
- 入力 x ∈ R^d, FFN の内側次元 d_ff。
- パラメータ W_a ∈ R^{d×d_ff}, W_b ∈ R^{d×d_ff}, and output projection W_o ∈ R^{d_ff×d}.
- FFN(x) = W_o^T ( Swish( x W_a ) ⊙ (x W_b) ).
- Swish(z) = z * sigmoid(z).

短い実装
```python
import numpy as np
def swiglu(x, Wa, Wb, Wo):
    # x: (d,), Wa/Wb: (d, d_ff), Wo: (d_ff, d)
    a = x.dot(Wa)        # (d_ff,)
    b = x.dot(Wb)        # (d_ff,)
    sw = a * (1 / (1 + np.exp(-a)))  # Swish
    return sw * b @ Wo   # (d,)
```

GQA（Query/Key-Value頭数の分離）を小さな数値例で詳述
直感：Queryを多く、Key/Valueを少なくすることで計算量を下げつつ、Query側の多様性を保つ。キー／値の頭が少ないと、複数のクエリ頭が同じKV頭を共有して「集約」する。内積を計算するために、クエリ側をグルーピングしてキーと内積の次元を合わせる。

設定：d_model = 8, query_heads h_q = 4, kv_heads h_k = 2。ここではシーケンス長を L=2 として数値計算をする（小さい値は説明用）。

定義と数値
- d_model = 8 は隠れ次元（学習パラメータ空間の幅）。
- 各 query head の深さ d_q = d_model / h_q = 8 / 4 = 2（学習パラメータから得られるactivationの次元）。
- 各 kv head の深さ d_k = d_model / h_k = 8 / 2 = 4。
- 比率 r = h_q / h_k = 2。GQAでは r 個の query head をまとめて d_k 次元になるよう連結して attention を取る。

具体的な数値例
- 入力系列 X の2時刻分を簡略化して与え、学習パラメータ W_Q, W_K, W_V を適当な小行列（ここでは整数）で取り計算する。
- 例として X = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]]（各ベクトルは長さ8）。
- W_Q を単位分割行列で設計すると、Q は各ヘッドに2成分ずつ割り当てられる。K は各ヘッドに4成分ずつ。

小表（要点）
| 項目 | 数値 |
|---:|:---|
|d_model|8|
|query heads h_q|4|
|kv heads h_k|2|
|d_q = d_model/h_q|2|
|d_k = d_model/h_k|4|
|ratio r = h_q/h_k|2|
|シーケンス長 L|2|

具体計算（抜粋）
1) Q = X @ W_Q → 得られる Q の各時刻は (h_q, d_q) = (4,2) にreshape。
2) K = X @ W_K → 得られる K は (h_k, d_k) = (2,4)。
3) Q を r=2 グループにして各グループの d_q を連結して長さ d_k = r * d_q = 4 にし、K と内積をとる。
4) softmax をかけて attention weights を得て、V と掛け合わせる。

簡易Python（数値を埋めて見せる）
```python
import numpy as np
X = np.array([[1,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0]], dtype=float)  # (L=2, d_model=8)
# W_Q: (8,8) choose to map to 4 heads of 2 dims each
W_Q = np.eye(8)
W_K = np.eye(8)
W_V = np.eye(8)
Q_full = X @ W_Q          # (2,8)
K_full = X @ W_K          # (2,8)
V_full = X @ W_V          # (2,8)

# reshape
Q = Q_full.reshape(2,4,2)   # (L, h_q, d_q)
K = K_full.reshape(2,2,4)   # (L, h_k, d_k)
V = V_full.reshape(2,2,4)   # (L, h_k, d_k)

# group r=2 Q heads -> (L, h_k, d_k)
Q_grp = Q.reshape(2,2,4)    # (L, h_k, r*d_q=d_k)
# attention at time t attending over times s: here compute t=0 attends s=0,1
scores = np.einsum('tad,sad->tas', Q_grp, K)  # (t over seq, a over h_k, s over seq)
# for simplicity, softmax over s:
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

attn = softmax(scores)  # (L, h_k, L)
out = np.einsum('tas,sad->tad', attn, V)  # (L, h_k, d_k)
print("Q shape", Q.shape, "K shape", K.shape, "attn shape", attn.shape, "out shape", out.shape)
```

FlashAttention（計算の融合とメモリ効率）
直感：Attention の計算は QK^T（大行列）→ softmax → 行列掛けという3段階で済むが、中間行列の全部をメモリに展開すると大きなコストになる。FlashAttention はこれらをタイル処理してソフトマックスを数値安定に逐次計算し、GPU メモリの読み書きを減らす。注意点：実装は複雑で、ハードウェア（GPUのキャッシュやワープ）に強く依存する。効果はシーケンス長やハードウェアで変わる。アルゴリズム上はソフトマックスの結合規則と分割可能性に依存する。

MoE（Mixture of Experts）の直感と注意点
直感：全てのサンプルに全てのパラメータを適用するのではなく、入力ごとに「どの専門家（サブネット）を使うか」を選んで一部の専門家のみを適用することで、理論上はパラメータを大幅に増やしつつ計算量は抑えられる。ルーティングは学習可能なゲート（小さな線形→softmax）で行う。

数式（簡略）
- 専門家が E 個、各専門家は同じ構造の FFN で、入力 x_i に対しゲート g(x_i) ∈ R^E を計算し、トップK（通常K=1か2）を選ぶ。
- 出力は選ばれた専門家の重み付き和。学習パラメータは専門家の重みとゲートの重み。
注意点：負荷分散（load balancing）や通信（分散訓練時に専門家が別デバイスにあると転送が発生）でトレーニングが難しい。実運用では容量因子（capacity factor）やAuxiliary lossでバランスを取る工夫が不可欠。

RoPE・RMSNorm・MoE の比較（目的／場所／注意点）
| 技術 | 目的 | Transformer 内の場所 | 注意点 |
|---|---|---:|---|
|RoPE|相対的な位置情報を線形に組み込み、外挿性を改善|Q/K の前（投影後に回転）|回転は線形で位相付与に過ぎず万能ではない|
|RMSNorm|軽量な正規化で数値安定と実装コスト低減|各サブレイヤの直前（LayerNorm の代替）|平均を引かないので平均シフトに注意|
|MoE|パラメータ増加と計算節約の両立|通常FFN 層を置換|負荷分散と通信、ルーティング学習が課題|

Transformer 内での前後処理・接続と典型的な誤解の訂正
- これらの技術は原理的に「元のTransformerに上書き」する形で使えるが、実運用では他の設計（学習率スケジュール、初期化、ドロップアウト、バッチサイズ）と絡んで動作が変わる。たとえば RMSNorm を単に LayerNorm と入れ替えれば学習が成功する保証はない。
- よくある誤解：Attention の重み行列の非ゼロ要素＝モデルが「注目」している完全な証拠。実際は重みは学習済みの線形変換群とスケールに依存し、局所的な確率分布の一面を見せるにすぎない。注意重みを直接解釈して人間の「注目」を断定するのは危険。
- 各技術の採用状況はモデル・用途・実装の制約で大きく異なる。ここで紹介したのはリアルな選択肢であって「全LLMの必須技術」ではない。

学習パラメータ vs 活性化の明確化
- 学習パラメータ（weights, g, W_a, W_b など）は訓練で更新される固定オブジェクト。
- 活性化（x, Q, K, V, out など）は入力とパラメータからその場で計算される値。上の数値例やコードで使った行列（W_*）は学習パラメータ、Q/K/V は活性化です。

章の終わりに：あなたが説明できること（チェックリスト）
- RoPE がなぜ相対位置情報の処理に向いているかとその数学的仕組み。
- RMSNorm が LayerNorm と何を変えるか（平均を引かないことの意味）。
- SwiGLU がどのようにFFNの表現力を高めるか。
- GQA の頭数非一致が計算のどの部分をどう変えるか（d_model=8, h_q=4, h_k=2 の例を再現できること）。
- FlashAttention と MoE がそれぞれどのシステム的問題を解決するかと実運用上の注意点。

確認問題（短め）
1) d_model=12, h_q=6, h_k=3 のとき d_q と d_k、比率 r はいくつか。r を使って Q のどの軸をどのように連結すれば K と内積できるか説明せよ。
2) RMSNorm の定義を書き、LayerNorm と比較して「平均項がない」ことが学習に与える可能性のある影響を一つ述べよ。

参考資料
- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [RoFormer — Su et al. (2021)](https://arxiv.org/abs/2104.09864)
- [RMSNorm — Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467)
- [GQA — Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245)
- [Sparsely-Gated Mixture-of-Experts — Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538)
- [FlashAttention-2 — Dao et al. (2023)](https://arxiv.org/abs/2307.08691)
- [PyTorch Embedding / CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

（注）本章の数値例は教材用に簡略化したもので、実運用の大規模モデル内部の値とは異なります。
