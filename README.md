# Transformer Atlas

**Transformerを、式の意味から理解する。**

Transformer Atlas は、Transformer を「用語集」ではなく**読める機械**として理解するための日本語インタラクティブ教材です。対象は、中学程度の計算を追える一方で、線形代数や微積を前提にされたくない成人学習者です。

本文は **直感 → 実際の機構 → 最小の数式 → 小さな例 → コード** を繰り返します。token、embedding、Q/K/V、softmax、causal mask、Transformer block、next-token prediction、loss、gradient、KV cache までを、最終的な小型 decoder-only Transformer と接続します。

## 第一版に含まれるもの

| 領域 | 内容 |
| --- | --- |
| コース | 7 Part・28章の連続カリキュラム。Part 概要、前後移動、概念索引を含む。 |
| 数学の横道 | vector、matrix、transpose、softmax、gradient など10ページ。前提コースではなく必要なときに参照する。 |
| 操作 | 教材用 tokenizer、dot product、softmax、causal mask のローカル実験。外部 API は不要。 |
| 実装 | attention を隠さない決定的な Tiny Transformer forward trace とテスト。 |
| 検証 | content/frontmatter、Python 実装、Astro 型検査・静的ビルド、内部リンク、CI。 |

## ローカルで動かす

```bash
pnpm install
pnpm dev
```

品質確認は次の順で実行します。

```bash
pnpm test
pnpm build
pnpm test:links
```

## 設計・出典・保守情報

- [カリキュラムと製品設計](./docs/DESIGN.md)
- [実行要件](./docs/MANUS_BRIEF.md)
- [出典と利用範囲](./docs/research/SOURCES.md)
- [サイトアーキテクチャ](./docs/ARCHITECTURE.md)
- [執筆・用語規約](./docs/CONTENT_STYLE.md)
- [配信手順](./docs/DEPLOYMENT.md)
- [既知の制約](./docs/KNOWN_LIMITATIONS.md)
- [貢献方法](./CONTRIBUTING.md)

## ライセンス

教材本文とプロジェクト文書は [CC BY 4.0](./LICENSE) です。各一次資料・外部ライブラリには、それぞれのライセンスが適用されます。
