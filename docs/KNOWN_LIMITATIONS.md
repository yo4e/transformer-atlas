# 既知の制約と次の改善候補

第一版は、Transformer の核心を自分で説明できるようになるための静的な土台を優先した。次の事項は意図的に未実装または簡略化している。

| 領域 | 現在の状態 | 改善候補 |
| --- | --- | --- |
| 数式組版 | 主要な式は code style の短い表記で示す | KaTeX を導入し、読み上げ情報を含む複雑な式を追加する。 |
| interactive atlas | tokenizer、dot product、softmax、mask の4部品 | Q/K/V の逐次 trace、sampling、KV cache、block flow を増やす。 |
| tokenizer | 教材用の局所分割であり本物の tokenizer ではない | ライセンスに適合する小さな vocabulary を同梱して BPE/Unigram の差を示す。 |
| tiny model | 決定的な forward trace。学習 loop と PyTorch 実装は省略 | 追加の明示的な training notebook と PyTorch 実装を併設する。 |
| 検索 | タイトルと概要のブラウザ内フィルタ。Pagefind index は生成済み | Pagefind UI を接続し、本文検索と日本語 query の確認を行う。 |
| 技術レビュー | 原著・一次資料に基づく初期レビュー | 各章への専門家レビューと、model-specific な出典の追記を進める。 |
