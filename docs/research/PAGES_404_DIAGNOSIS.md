# GitHub Pages 404 診断

2026-08-15 に公開サイト `https://yo4e.github.io/transformer-atlas/` を確認した。GitHub Pages の公開設定と直近のデプロイは正常であり、トップページと、リポジトリ名を含む代表章URLは HTTP 200 を返した。一方、公開された HTML 内のナビゲーションと章リンクが `/chapters/...`、`/parts/...`、`/math/...` のような**ドメイン直下の絶対パス**になっていた。

プロジェクトサイトは `/transformer-atlas/` を基底パスとして公開されるため、このリンクをクリックすると `https://yo4e.github.io/chapters/...` を要求して404になる。正しいURLは `https://yo4e.github.io/transformer-atlas/chapters/...` である。したがって、根本原因はデプロイ失敗ではなく、Astro の静的出力がGitHub Pagesのリポジトリサブパスを考慮していないことである。

| 確認項目 | 結果 |
| --- | --- |
| Pages設定 | workflow公開、HTTPS有効 |
| トップURL | HTTP 200 |
| サブパス付き代表章URL | HTTP 200 |
| HTML内のリンク | リポジトリ名を欠く絶対パス。修正が必要 |

修正では Astro の `base` を `/transformer-atlas` に設定し、生成済みの内部リンク・アセット参照がこの基底パスを自動的に含むようにする。

リンク監査では、レイアウト、トップ、Part、概念索引、検索、章・数学ページ、本文中の相互参照に、`/chapters/...`、`/parts/...`、`/math/...` の形式が残ることを確認した。Astro の `base` 設定は静的公開の基底パスを与えるが、テンプレートとMarkdownで明示したドメイン直下への絶対リンクを自動変換しない。このため、ビルド後にすべての内部リンクを基底パス付きへ正規化する検査・変換を加える。
