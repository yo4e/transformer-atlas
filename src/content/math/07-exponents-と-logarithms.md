---
title: "Exponents と logarithms"
order: 7
summary: "softmax と loss に現れる二つの逆向きの演算"
concepts: ["Exponents と logarithms"]
references:
  - label: "Vaswani et al. (2017), Attention Is All You Need"
    url: "https://arxiv.org/abs/1706.03762"
---

# Exponents と logarithms

**なぜTransformerのどの箇所で必要か（要点）**

Transformer（Vaswani et al., 2017）でsoftmaxと負の対数（negative log）は頻繁に登場します。代表的には（1）自己注意機構の重み計算でスコアに対してsoftmaxを取ることで確率分布に変換し、重み付き和を作る点、（2）言語モデルや分類タスクで出力ロジットを確率に変換して正解ラベルに対する損失（クロスエントロピー＝期待負の対数尤度）を計算する点、の二箇所です。ここでは数学を最小限にして、exp と log が互いに「逆の役割」を持つという観点だけを導入します。元のTransformerは Vaswani et al. 2017 によるもので、後年の変種（縮小・正規化の工夫や損失関数の拡張など）はここで明示的に区別します。

**最小の定義（必要最小限の式）**

- 指数関数（natural exponential）: exp(x) = e^x。正の実数を返す。
- 対数（natural logarithm）: log(p) = ln(p)。p>0 のとき exp(log(p)) = p、log(exp(x)) = x。
- softmax（ベクトル s の要素を確率にする）:
  p_i = exp(s_i) / Σ_j exp(s_j)
  ここで Σ_i p_i = 1 かつ p_i > 0。
- negative log（確率を損失に変える）:
  L(p) = −log(p)
  確率が大きいほど損失は小さくなる。

softmaxは「ログ領域にあるスコア s を正規化して確率にする操作」、negative logは「確率を取り、それを0以上の損失値に変換する操作」です。両者は向き合う役割を持ちますが、厳密には互いの逆関数ではありません（後述）。

**数値例（手順を省略せず計算）**

与えられたスコア列 s = [0, 1, 2] について softmax を計算します。自然指数 e を用い、段階的に示します。

1) 指数を取る:
   e^0 = 1
   e^1 ≈ 2.718281828459045
   e^2 ≈ 7.38905609893065

2) 合計を取る:
   Σ = 1 + 2.718281828459045 + 7.38905609893065 ≈ 11.107337927389695

3) 各要素を合計で割って正規化（softmaxの確率）:
   p0 = 1 / 11.107337927389695 ≈ 0.09003057317038046
   p1 = 2.718281828459045 / 11.107337927389695 ≈ 0.24472847105479764
   p2 = 7.38905609893065 / 11.107337927389695 ≈ 0.6652409557748218

確認: p0 + p1 + p2 ≈ 1（厳密に 1 になるはずだが丸め誤差あり）。

次に確率 0.8 と 0.2 に対する負の対数を計算します（自然対数を使用）。

- L(0.8) = −ln(0.8) ≈ −(−0.2231435513142097) = 0.2231435513142097
- L(0.2) = −ln(0.2) ≈ −(−1.6094379124341003) = 1.6094379124341003

解釈: 0.8 の確率を与えられたときの損失は約0.223、0.2 のときは約1.609。確率が高いほど損失は小さい。

**記号と shape の解読**

- s: ロジット（スコア）ベクトル。通常 R^K（K は選択肢や語彙の数）。
- softmax(s): 同じ形状 R^K の確率ベクトル。各要素は正で総和が1。
- バッチや系列を扱うときは形状が拡張される。例:
  - 分類出力のロジット: logits の shape = (batch_size, K)
  - 言語モデルの単語予測: logits shape = (batch_size, seq_len, vocab_size)
  - 注意（attention）のスコア: raw_scores shape = (batch_size, n_heads, seq_len, seq_len)
  softmax は最後の適切な軸に沿って適用される（例: attention では”キー”方向に沿って正規化して重みを得る）。

注意すべき点: ロジットから確率への写像はsoftmax。確率 p から元のロジット s を完全に一意に復元することはできませんが、log p = s − logsumexp(s) の関係から、ロジットは定数を足し引きしても同じ確率を与える（つまりロジットは定数オフセットについて不定）。

**数値安定性の短い注意（実用上重要）**

exp は大きな正の引数でオーバーフローし、小さな負の引数でアンダーフローに弱い。実運用では softmax を計算する前に最大値を引くことで安定化するのが標準手法：
  softmax(s) = softmax(s − max(s))
この操作は分母と分子に同じ因子 e^{−max(s)} を掛けるのと同値で、確率は変わりません。Transformer の論文でもスケール（√d_k）や安定化の議論が重要です（Vaswani et al., 2017）。

**透明な短い Python コード（実行可能）**

```python
import math
import numpy as np

s = np.array([0.0, 1.0, 2.0])

# naive softmax（説明用）
exp_s = np.exp(s)
sum_exp_s = exp_s.sum()
p_naive = exp_s / sum_exp_s

# 安定化した softmax（実務で推奨）
s_stable = s - s.max()
exp_s_stable = np.exp(s_stable)
p_stable = exp_s_stable / exp_s_stable.sum()

# negative log の例
p_a = 0.8
p_b = 0.2
loss_a = -math.log(p_a)
loss_b = -math.log(p_b)

print("naive softmax probabilities:", p_naive)
print("stable softmax probabilities:", p_stable)
print("loss for p=0.8:", loss_a)
print("loss for p=0.2:", loss_b)
```

このコードは上で示した数値を生みます。安定化した softmax は naive な実装と同じ結果（理論上）を返しますが、浮動小数点の扱いで差が出る状況を防げます。

**よくある誤解と正しい理解**

- 誤解: softmax の後に log を取れば元のスコアが復元できる。
  - 正: log(p_i) = s_i − logsumexp(s)。s_i に一定の定数を足しても p は変わらないため、p から元の s を一意に決められない。復元できるのは「相対的な差」だけ（定数オフセットは不定）。
- 誤解: negative log は単なる符号反転で意味はない。
  - 正: negative log は確率を「損失」に変換する自然な方法で、尤度最大化と損失最小化を結びつける。学習において微分しやすく、確率が高いほど損失が小さくなる単調性を持つ。
- 誤解: softmax は線形変換のように振る舞う。
  - 正: exp は非線形（掛け算に変換する）ので、softmax はロジット間の差に敏感で、値のスケーリングが結果に大きな影響を与える。だからTransformerではスコアを√d_kでスケーリングする（Vaswani et al., 2017）。

**本文の関連章へのリンク**

- 数値計算の安定性（実装上の工夫）: https://en.wikipedia.org/wiki/LogSumExp
- softmax と交差エントロピーの関係（数学的詳細）: https://en.wikipedia.org/wiki/Cross_entropy
- Transformer（原著）: https://arxiv.org/abs/1706.03762

（注）上は参考リンクです。Transformer に関する事実は Vaswani et al. 2017 の原著に基づきます。後年の改良や変種（例: 正規化・重み共有・損失関数の拡張など）は個別に文献を確認してください。

**読者が説明できること（学習ゴール）**

- softmax がどのようにロジットを確率に変換するかを説明できる。
- exp と log が互いにどう関係し、なぜ negative log が損失として意味を持つのか説明できる。
- 与えられた具体例 s = [0,1,2] から確率を手計算（あるいはコードで）して結果を解釈できる。
- 確率 0.8 と 0.2 に対する -ln の値を示し、その意味（より高い確率はより小さな損失）を説明できる。
- 実装上の数値安定化（max を引く）を説明し、なぜ必要か例を挙げて説明できる。

**参照**

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. arXiv:1706.03762. https://arxiv.org/abs/1706.03762
- LogSumExp の解説（数値安定化）: Wikipedia. https://en.wikipedia.org/wiki/LogSumExp
- Cross-entropy の解説: Wikipedia. https://en.wikipedia.org/wiki/Cross_entropy

この短いノートは、Transformer の主要部分で何が行われているかを理解するために必要な exp と log の関係に集中しました。手計算を速く行う訓練は不要ですが、数が何を表すか（スコア、確率、損失）を常に意識しておくことが重要です。
