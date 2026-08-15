# Contributing to Transformer Atlas

貢献を歓迎します。技術的な変更は、まず [Issue](https://github.com/yo4e/transformer-atlas/issues) で目的と出典を共有してください。

教材本文は日本語で書き、概念を **問題 → 直感 → 機構 → 最小の数式 → 小さな例 → コード → 誤解しやすい点** の順に説明します。一次資料の文章・図を転載せず、独自の説明と図を作ってください。新しい技術主張には、`docs/research/SOURCES.md` に出典と利用範囲を追加します。

## ローカル確認

```bash
pnpm install
pnpm test
pnpm build
```

対話部品を変更する場合は、キーボード操作・フォーカス・reduced motion・狭い画面での表示も確認してください。
