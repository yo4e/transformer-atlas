# サイトアーキテクチャ

Transformer Atlas は **Astro による静的サイト** である。本文は `src/content` に置き、ルート・レイアウト・対話部品は `src/pages`、`src/layouts`、`src/components` に分ける。サーバー、データベース、ログイン、必須の外部 API は使わない。

| 層 | 場所 | 責務 |
| --- | --- | --- |
| 教材ソース | `src/content/chapters`, `src/content/math` | frontmatter を持つ MD ソース。表示から分離し、将来の PDF/EPUB 変換に残す。 |
| スキーマ | `src/content.config.ts` | 章番号、Part、概念、参照、公開状態を検証する。 |
| 画面 | `src/pages` | ホーム、章、数学索引、Part、概念索引、静的検索を生成する。 |
| 表現 | `src/layouts`, `src/styles` | Marginalia Atlas の読書面、ナビゲーション、注釈を担う。 |
| 操作 | `src/components` | tokenizer の教材用分割、dot product、softmax、causal mask のローカル実験。 |
| 検証 | `scripts`, `examples`, `.github/workflows` | 内容件数、Tiny Transformer、Astro、内部リンク、静的ビルドを確認する。 |

対話部品はすべてブラウザだけで動作する。入力内容を送信せず、教育用の小さな計算・明示的なラベル・テキスト説明をセットにする。生成物 `dist/` は GitHub Pages または Cloudflare Pages にそのまま配信できる。
