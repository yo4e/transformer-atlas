# 配信手順

## GitHub Pages

このリポジトリには `.github/workflows/deploy-pages.yml` が含まれる。リポジトリ設定の **Settings → Pages → Build and deployment → Source** で **GitHub Actions** を選ぶと、`main` への push または手動実行で静的サイトを公開できる。

`astro.config.mjs` は GitHub Actions 実行時に `/transformer-atlas` を base path として使う。フォークまたはリポジトリ名を変える場合は、この値を変更する。

## Cloudflare Pages

Cloudflare Pages では、Node 22 と pnpm を使い、次の設定でよい。

| 設定 | 値 |
| --- | --- |
| Build command | `pnpm install --frozen-lockfile && pnpm build` |
| Build output directory | `dist` |
| Node.js version | `22` |

Cloudflare Pages 用に公開する場合は、`astro.config.mjs` の `base` を空にするか、独自の配信パスに合わせて変更する。どちらのホストでも、追加の API key・データベース・課金サービスは不要である。
