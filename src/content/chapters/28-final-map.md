---
title: "Final map"
part: 7
order: 28
summary: "全体地図に戻り、各部品を接続する"
prerequisites: []
concepts: ["Final map"]
interactive_components: []
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
status: complete
---

# Final map

**問題提起 — なぜ段階的な地図が必要か**

言語モデルは「文字列を次の記号に変換する」だけに見えますが、内部では複数の役割が分担された段階を通ります。生のテキスト（"I like apples"）がただちに次の語を出すわけではなく、まず符号化（ID化）、次に数値ベクトル化、内部演算（attention と feed‑forward の繰り返し）を経て、最後にカテゴリ（語彙上の確率分布）に戻されます。これらの段階を明確に区別し、その都度「何が問題で」「どの演算が行われ」「出てくるオブジェクトのshapeと数値はどうなるのか」を一貫して追える地図が学習・デバッグ・説明に不可欠です。ここでは「text → IDs → vectors → blocks → logits → selection」という矢印ごとに具体的な演算と役割を示し、3トークンの数値例を最後まで計算します。

（注意）この章で使う小さな実数例は教材的に単純化したもので、実際の大規模モデルの内部値を反映するものではありません。また「token」は必ずしも英単語と同義ではありません。attention の重みを人間的な“注目”の完全な証拠として扱うべきではない点も後で述べます。

直感とその限界

直感：テキストはまず離散IDに直され、IDは埋め込み行列で連続ベクトルに写像される。ベクトル同士の線形代数（内積や線形変換）がおもに計算を担い、最後に線形層＋softmaxで確率に戻す――という流れです。限界：attention の「大きな重み＝重要」は必ずしも解釈可能性を保証しません。重みはモデルの内部線形空間における数値最適化の結果であり、人間の意味的注目とは別物です。

用語の区別（重要）
- 学習されるパラメータ（parameters）：埋め込み行列 E、クエリ/キー/バリューの重み Wq, Wk, Wv、出力投影 Wo、FFN の重み W1, W2、（LayerNorm のスケール等）。これらは学習で更新される固定オブジェクト。
- 活性（activations）：ある入力に対して計算される中間ベクトル（埋め込みベクトル、Q/K/V、attention 出力など）。逐次的に変化する一時的データ。
- 軸（axes）：シーケンス軸（長さ L）、埋め込み次元（d_model）、ヘッド軸（h）、バッチ軸（B）。行列積でどの軸がどの役割かを常に意識する。

3トークンの小さな数値例（最初から最後まで）

設定（簡潔化した GPT 型、1 層、1 ヘッド、埋め込み次元 d=4、語彙サイズ V=10、因子の簡略化のため Wq=Wk=Wv=I を採用）
- テキスト（人間向け表現）: "T0 T1 T2"（実際はサブワード等）
- IDs: [3, 7, 2]
- 埋め込み行列 E (V×d) の該当行のみ示す（他はゼロとする簡略化）
  - E[3] = [0.5, 0.1, -0.3, 0.0]
  - E[7] = [0.0, 0.2, 0.1, 0.4]
  - E[2] = [0.3, -0.2, 0.0, 0.1]
- 位置埋め込み P[0..2]:
  - P[0] = [0.01, 0.02, 0.03, 0.04]
  - P[1] = [0.02, -0.01, 0.01, 0.0]
  - P[2] = [0.0, 0.01, -0.02, 0.03]
- d_k = 4, スケーリング係数 sqrt(d_k)=2
- クエリ/キー/バリュー重み Wq=Wk=Wv=I（説明の簡便のため）、Wo=I、FFN は簡略化して W1=0.5·I、非線形はReLU、W2=I とする。
（再掲）これらの小さな数値は教材用であり実モデルの値ではありません。

1) text → IDs
- 問題：語彙表にない文字列はどうするか、連続文字列をどう分割するか（トークナイザーの役割）。
- 今回は既に IDs = [3,7,2] が与えられているものとします。

2) IDs → vectors（埋め込み）
- 演算：X_i = E[id_i] + P[i]
- 具体計算（各ベクトルは長さ4）
  - X0 = E[3] + P0 = [0.5+0.01, 0.1+0.02, -0.3+0.03, 0+0.04] = [0.51, 0.12, -0.27, 0.04]
  - X1 = E[7] + P1 = [0.02, 0.19, 0.11, 0.4]
  - X2 = E[2] + P2 = [0.3, -0.19, -0.02, 0.13]
- ここで E は学習パラメータ、X は活性です。shape：X は (L=3, d=4)。

3) vectors → blocks（自己注意 + 残差 + FFN）
- 自己注意（Scaled Dot‑Product, causal mask）
  - Q = X·Wq = X（今回は Wq=I）
  - K = X·Wk = X
  - V = X·Wv = X
  - スコア S[i,j] = (Q_i · K_j) / sqrt(d_k)
  - causal mask で j>i は -inf にして softmax に含めない（生成では未来を見ないため）
- 計算（内積の一部を示す）
  - 例 Q0·K0 = 0.51^2 + 0.12^2 + (-0.27)^2 + 0.04^2 = 0.349
  - Q0·K1 ≈ 0.0193, Q0·K2 ≈ 0.1408, など（計算は本文中に列挙）
  - スケーリングで S0 = [0.1745, 0.00965, 0.0704] など
  - causal mask を適用して行ごとに softmax：
    - row0（位置0）は [0.1745] → attention probs [1.0]
    - row1（位置1）は softmax([0.00965, 0.1043]) ≈ [0.4765, 0.5235]
    - row2（位置2）は softmax([0.0704, 0.00985, 0.0717]) ≈ [0.3399, 0.3200, 0.3401]
- attention 出力 O_i = sum_j prob[i,j] * V_j
  - O0 = X0
  - O1 ≈ [0.25247, 0.15665, -0.07107, 0.22846]
  - O2 ≈ [0.28178, 0.03697, -0.06338, 0.18581]
- 残差結合（簡略）：Y = X + O（通常は LayerNorm を挟む）
  - Y0 = [1.02, 0.24, -0.54, 0.08], など
- FFN（簡略版）: U = ReLU(W1·Y + b1)（W1=0.5·I），出力 = Y + W2·U（残差）
  - U ≈ ReLU(0.5·Y)
  - 最終出力 Z ≈ Y + U
  - Z0 ≈ [1.53, 0.36, -0.54, 0.12], Z1 ≈ [0.4087, 0.5200, 0.0584, 0.9427], Z2 ≈ [0.8727, -0.1530, -0.0834, 0.4737]
- ここまでで「blocks」の出力は Z（shape (3,4)）。Wq/Wk/Wv/Wo/W1/W2 や埋め込み E は学習パラメータ、Q,K,V,S,O,Y,U,Z は活性です。

4) blocks → logits
- 問題：ベクトルを語彙上のスコアに戻す。方法は語彙に対する線形写像。GPT 系では埋め込み行列 E を転置して共有することが一般的（tied embeddings）。ここでは簡略のため E をそのまま用いる。
- 演算：logits_i = Z_i · E^T（各語彙 k について dot(Z_i, E[k])）
- 具体（Z2 を用いて次のトークンを予測する例）
  - dot(Z2, E[3]) ≈ 0.4460
  - dot(Z2, E[2]) ≈ 0.3398
  - dot(Z2, E[7]) ≈ 0.1505
  - それ以外の語彙は E をゼロにした簡略モデルなので logits = 0
- これらは「生のスコア」であり確率ではない。

5) logits → selection（softmax とサンプリング／argmax）
- 演算：p = softmax(logits)（語彙全体で指数化し正規化）
- 例（V=10 のうち上で示した3つ以外のlogit=0とした場合）
  - exp(0.446)=1.562, exp(0.340)=1.405, exp(0.151)=1.163, その他7個は exp(0)=1
  - 合計 ≈ 11.13、確率は id=3 が約0.140、id=2 が約0.126、id=7 が約0.104、その他各0.089
- 選択：次のトークンは argmax（確率最大）または温度付きサンプリング／トップK／トップP などでサンプリングされる。生成時の振る舞いはこの選択規則（selection policy）に依存する。

短い Python（NumPy）実装例（変数名は数式と対応）
```python
import numpy as np

E = np.zeros((10,4))
E[3] = np.array([0.5,0.1,-0.3,0.0])
E[7] = np.array([0.0,0.2,0.1,0.4])
E[2] = np.array([0.3,-0.2,0.0,0.1])
P = np.array([[0.01,0.02,0.03,0.04],
              [0.02,-0.01,0.01,0.0],
              [0.0,0.01,-0.02,0.03]])

ids = [3,7,2]
X = np.stack([E[i] for i in ids]) + P  # (3,4)

# attention (Wq=Wk=Wv=I)
Q = K = V = X
scores = Q @ K.T / 2.0  # (3,3)
# causal mask: set upper triangle (j>i) to -1e9
mask = np.triu(np.ones_like(scores), k=1).astype(bool)
scores[mask] = -1e9
probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
probs = probs / probs.sum(axis=1, keepdims=True)

O = probs @ V  # (3,4)
Y = X + O
U = np.maximum(0.5 * Y, 0.0)  # ReLU(W1 Y), W1 = 0.5 I
Z = Y + U

# logits via tied embeddings E^T
logits = Z @ E.T  # (3,10)
# softmax for last position
last_logits = logits[-1]
p = np.exp(last_logits) / np.exp(last_logits).sum()
print("probabilities for vocab indices 0..9:", np.round(p,4))
```

全体構造の前後関係と実システムへの接続

- 実際のフル Transformer（GPT 型）では上のブロックが多数（数十〜数百）積み重なり、各ブロックで Wq/Wk/Wv/Wo は異なる学習パラメータです。微妙だが重要な点：LayerNorm の位置（pre‑LN vs post‑LN）、利用する非線形（ReLU vs GELU）、ヘッド数やヘッド内の d_k は実装で違う。ここでの単一ヘッド・単層は理解を助けるための簡略化。
- 学習時は損失関数（通常は交差エントロピー）で logits と正解トークンの one‑hot を比較し、誤差逆伝播で上記パラメータを更新する。推論時は causal mask に従って1トークンずつ生成する手続き（キャッシュを使って計算効率化）を行う。
- 現実の大規模モデルでは埋め込み共有や正規化の扱い、並列化（データ並列／モデル並列）、高速化ライブラリ（FlashAttention）や sparsity（Mixture of Experts）など多くの実装上の工夫が追加されるが、基本的な「text→IDs→vectors→blocks→logits→selection」は変わらない。

ありふれた誤解の訂正

- 「attention の重みが高い箇所＝モデルがそこに“注目している”＝その理由を説明できる」：注意重みは内部の線形演算によるスコアであり、必ずしも意味論的説明と同一視できない。解釈には補助的な解析手法や因果的検証が必要です。
- 「token = word」：実装上の token は BPE や SentencePiece による細分化単位であり、単語より細かいことが多い。ID と語の1対1対応は存在しない。

学習パラメータと活性のまとめ（読み下し）
- 学習パラメータ例：E (V×d)、P (L×d)（位置埋め込みが学習される場合）、Wq,Wk,Wv (d×d 各ヘッド)、Wo (d×d)、FFN の W1 (d×d_ff)、W2 (d_ff×d)、LayerNorm のスケール・シフトなど。
- 活性例：X（入力埋め込み）、Q,K,V（クエリ/キー/バリュー）、S（スコア行列）、probs（attention 確率）、O（attention 出力）、Y,Z（残差後の出力）、logits（語彙スコア）。

この章を学んだあとに説明できること（短いチェックリスト）
- text→IDs→vectors→blocks→logits→selection の各段階で「何を入力にし、何を出力するか」を言える。
- attention のスコア計算（内積→スケーリング→mask→softmax→加重和）の数式と、軸（sequence, feature, head）がどこにあるか説明できる。
- 学習されるパラメータと一時的な活性を区別できる。
- 小さな具体例で実際に数値を手で追える（ここで示した3トークン例）。

確認問題（短め、解答は自分で試す）
1. 先ほどの 3 トークン例で、もし Wv を 2·I（V にかかる係数が2倍）に変更したら、最終 logits に与える直接的な影響はどの変数を通じて起きますか？（経路を説明し、符号での増減を述べよ）
2. causal mask を外して全結合 attention を行った場合（すべての j にアクセス可能）、位置1（中間位置）の attention 確率はどう変わるか。具体的にどのようなスコア行列の項が追加されるかを述べ、同時に生成時に引き起こされる問題点を説明せよ。

参考文献
- Vaswani, A., et al. (2017), "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- Radford, A., et al. (2019), "Language Models are Unsupervised Multitask Learners": https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- PyTorch: Embedding / CrossEntropyLoss ドキュメント（実装参照）: https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html

（補足）次に学ぶ道筋の例
- attention を複数ヘッド化すると何が変わるか（head 軸の導入とパラメータ分割）
- LayerNorm の前置／後置（pre‑LN vs post‑LN）と学習安定性
- トークナイザーと語彙設計（BPE, SentencePiece）の実務的影響
これらは本章の地図を使えば自然に学べます。
