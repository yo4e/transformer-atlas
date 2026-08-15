# 出典と利用範囲

Transformer Atlas の説明は、原著論文・一次資料・公式ドキュメントを起点とし、本文は独自に執筆する。以下の表では、出典が支える範囲と、本文で区別するべき事実の種類を明示する。

| ID | 出典 | 支える主な内容 | 利用範囲 |
| --- | --- | --- | --- |
| S1 | Vaswani et al., *Attention Is All You Need* | original Transformer、scaled dot-product attention、multi-head attention、sinusoidal positional encoding、encoder–decoder、masking | 第1・5–16章、数式参照 |
| S2 | Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* | subword units と BPE を使う語彙外問題への対処 | 第2章 |
| S3 | Radford et al., *Language Models are Unsupervised Multitask Learners* | GPT-2 の decoder-only language modeling、byte-level BPE | 第14–17章、22–23章 |
| S4 | Su et al., *RoFormer* | RoPE が query/key に位置を回転として組み込み、相対位置依存を注意式に含めること | 第10・24章 |
| S5 | Zhang & Sennrich, *RMSNorm* | RMSNorm の定義、LayerNorm に対する単純化と設計意図 | 第12・24章 |
| S6 | Ainslie et al., *GQA* | MQA は1組の K/V head、GQA は query head 数より少なく1より多い K/V head を用いる一般化 | 第23・24章 |
| S7 | Shazeer et al., *Sparsely-Gated MoE* | router が入力ごとに sparse な expert 組合せを選ぶ条件付き計算 | 第24章 |
| S8 | Dao et al., *FlashAttention-2* | attention 実装の計算・メモリ効率を改善する後年の実装技術 | 第22・24章 |
| S9 | PyTorch 公式ドキュメント | `Embedding` が語彙の lookup table として使えること、`CrossEntropyLoss` の logits と目標カテゴリの関係 | 第3・19・26章の実装注記 |

> **記法上の約束:** S1 の original Transformer の事実は「原著」、S2–S5 の設計は「近年の一例」、教材内の小さな数値・図・トークン分割は「教育用の簡略化」と明示する。特定の商用モデルについて、一次資料にない採用・非採用を断定しない。

## 参照先

1. [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [Sennrich et al. (2016), Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
3. [Radford et al. (2019), Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
4. [Su et al. (2021), RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
5. [Zhang & Sennrich (2019), Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
6. [Ainslie et al. (2023), GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
7. [Shazeer et al. (2017), The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
8. [Dao et al. (2023), FlashAttention-2](https://arxiv.org/abs/2307.08691)
9. [PyTorch: Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) and [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
