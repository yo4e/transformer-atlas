import { mkdir, writeFile } from "node:fs/promises";

const source = {
  label: "Vaswani et al. (2017), Attention Is All You Need",
  url: "https://arxiv.org/abs/1706.03762"
};

const chapters = [
  [1, 1, "Transformer は何をしているのか", "文字列が次の token の分布になるまでの全体地図", "text → token ID → embedding → Transformer block → logits → distribution", "入力を次の token 候補へ変換する道筋を、部品名だけでなく情報の形で追う。"],
  [1, 2, "Tokens: 文字列が部品になる", "文字列を token ID に変える tokenizer の役割", "text → tokenizer → token IDs", "モデルは文字そのものを読まない。tokenizer が決めた分割と ID を受け取る。"],
  [1, 3, "Embeddings: ID がベクトルになる", "辞書の行を引いて token を連続値ベクトルにする", "X = Embedding[token_ids]", "ID の大小は意味の近さではない。embedding table の行を引くことで初めて計算できる表現になる。"],
  [1, 4, "Dot product: 向きと関連度を比べる", "対応する要素を掛けて足す比較", "a · b = Σ a_i b_i", "dot product は二つの同じ長さのベクトルを一つの score に縮約する。attention では関連度の材料になる。"],
  [2, 5, "Attention が解く問題", "ある位置が他の位置から必要な情報を集める仕組み", "output_i = Σ_j weight_ij value_j", "各位置が同じ固定窓だけを見るのでなく、入力ごとに重みを変えて情報を集められる必要がある。"],
  [2, 6, "Query, Key, Value", "学習済み線形射影で、探す側・照合側・運ぶ情報を作る", "Q=XW_Q, K=XW_K, V=XW_V", "Q/K/V は人間の検索比喩ではなく、同じ入力 X に別の学習済み行列を掛けた数値ベクトルである。"],
  [2, 7, "Scaling と softmax", "score を比較可能な重みに変える", "weights = softmax(scores / √d_k)", "softmax は行ごとの相対重みを正にして和を1にする。値は自動的に確信度を意味しない。"],
  [2, 8, "Self-attention を最後まで通す", "score matrix から weighted sum までを一つの式で結ぶ", "Attention(Q,K,V)=softmax(QKᵀ/√d_k)V", "左から右へ、各行が『今の位置が集めるべき value の混合比』を作る。"],
  [2, 9, "Multi-head attention", "複数の投影空間で並行して attention を計算する", "MHA(X)=Concat(head₁,…,head_h)W_O", "head はあらかじめ文法や感情を担当する箱ではない。異なる射影を学習する並列計算である。"],
  [3, 10, "Position: 順序を入れる", "content-only attention に位置情報を渡す", "X₀ = embedding + position encoding", "原著の sinusoidal encoding と、Q/K を回転して相対位置を扱う RoPE は異なる選択肢である。"],
  [3, 11, "Feed-forward network", "位置間を混ぜた後、各位置を独立に変形する", "MLP(x)=W₂ activation(W₁x+b₁)+b₂", "attention が位置をまたいで混ぜた情報を、MLP は各位置で非線形に再表現する。"],
  [3, 12, "Residual connection と normalization", "情報経路を保ち、層を安定して重ねる", "y = x + sublayer(norm(x))", "residual path は入力を足し戻す経路、normalization は値のスケールを扱いやすくする操作である。"],
  [3, 13, "一つの Transformer block", "attention と MLP を残差経路でつないだ繰り返し単位", "x → norm → attention → +x → norm → MLP → +", "block は魔法の箱ではない。ここまでに見た演算を一定の順序で合成したものだ。"],
  [4, 14, "Encoder, decoder, decoder-only", "original Transformer と GPT 型の見通しを分ける", "encoder: bidirectional; decoder: causal", "original Transformer は encoder–decoder 構成で翻訳を扱った。decoder-only は過去だけを見て次 token を予測する。"],
  [4, 15, "Causal masking", "未来の token を見ずに学ぶための三角形の制約", "score_ij = −∞ when j > i", "training で答えの先読みを防ぐ。mask は token を消すのでなく、許されない score を softmax の前に除外する。"],
  [4, 16, "Logits と次 token 予測", "最終ベクトルを語彙ごとの未正規化 score に変える", "logits = hW_vocabᵀ", "logit は token ごとの生の score で、softmax を通す前は確率でも順位表でもない。"],
  [4, 17, "Temperature, top-k, top-p", "出力分布から token を選ぶ規則", "p = softmax(logits / temperature)", "temperature は logits の差を縮小・拡大する。top-k/top-p は候補集合を切ってから再正規化する。"],
  [5, 18, "何が学習されるのか", "architecture と parameter の内容を分けて考える", "θ = {embeddings, W_Q, W_K, W_V, …}", "層の接続図は architecture、値が更新される行列やベクトルは parameter である。"],
  [5, 19, "Loss: 間違いを数値にする", "正解 token に割り当てた確率を損失にする", "loss = −log p(target)", "正解の確率が小さいほど loss は大きい。これは次 token の分類問題における学習信号である。"],
  [5, 20, "Gradient と backpropagation", "parameter をどちらへ動かせば loss が減るかを局所的に測る", "θ ← θ − learning_rate × ∇θ loss", "gradient は一つの微小な変化が loss をどちらへどれほど動かすかの近似である。"],
  [5, 21, "Training loop", "forward・loss・backward・update を繰り返す", "batch → forward → loss → backward → update", "一回の更新で知識が完成するのではない。多くの batch で parameter を少しずつ変える反復である。"],
  [6, 22, "Context window", "一度に扱える token の範囲と attention の計算量", "score shape: sequence_length × sequence_length", "context は会話の意味ではなく、その forward pass に入力した token 列の長さという操作的な制約でもある。"],
  [6, 23, "KV cache", "生成済み token の K/V を再利用して推論を短縮する", "reuse K₁…t, V₁…t for token t+1", "自己回帰生成で過去 token の K/V は変わらない。cache は同じ計算を毎回やり直さないための保存である。"],
  [6, 24, "Modern variations", "core Transformer の周囲にある代表的な設計選択", "RoPE / RMSNorm / GQA / MoE", "近年の設計は目的も採用状況も異なる。ここでは一つの『現代標準』として混ぜず、各手法の問題設定を分ける。"],
  [6, 25, "Transformer だけでは説明できないこと", "architecture と data・post-training・tools を混同しない", "capability ≠ architecture alone", "Transformer は計算骨格の一部である。訓練データ、目的関数、後学習、検索、ツール、agent 制御は別の層にある。"],
  [7, 26, "小さな Transformer を組み立てる", "attention を隠さない decoder-only 実装", "embedding → causal attention → MLP → logits", "この章では `examples/tiny_transformer.py` の小さな前向き計算を読む。production 用の高速化ではなく、shape を追うための教材である。"],
  [7, 27, "一つの token を追跡する", "token ID から logits までの tensor shape を検査する", "[T] → [T,d_model] → [T,T] → [T,d_model] → [T,vocab]", "実装の各中間値を表示して、式とコードの対応を一つずつ確認する。"],
  [7, 28, "Final map", "全体地図に戻り、各部品を接続する", "text → IDs → vectors → blocks → logits → selection", "最初に見た地図は、今なら各矢印がどの演算で、何の shape を持つかまで説明できる。"]
];

const partNames = ["テキストからベクトルへ", "Attention", "Transformer block", "Transformer から GPT へ", "学習", "現代の LLM を動かす", "一つ作る"];

function questions(title) {
  return `1. 「${title}」で、入力と出力はそれぞれ何か。\n2. この計算がなければ、どの問題が残るか。\n3. 教材の小さな例と実際の大規模モデルで、同じ点と異なる点は何か。`;
}

function yaml(value) {
  return JSON.stringify(value);
}

function body([part, order, title, summary, formula, mechanism]) {
  return `---\ntitle: ${yaml(title)}\npart: ${part}\norder: ${order}\nsummary: ${yaml(summary)}\nprerequisites: []\nconcepts: [${yaml(title)}]\ninteractive_components: []\nreferences:\n  - label: ${yaml(source.label)}\n    url: ${yaml(source.url)}\nstatus: complete\n---\n\n# ${title}\n\n## 問題\n\n${summary}。この章では、全体のどこでこの部品が必要になり、何を受け取り、何を返すかを先に固定する。\n\n## 直感\n\n${mechanism}\n\n## 実際の機構\n\nここで扱う計算を短く書くと、\`${formula}\` となる。記号は小さな例で確認してから使う。教育用の図と数値は概念を追うために小さくしており、実モデルの内部状態をそのまま可視化したものではない。\n\n## 最小の数式と shape\n\nベクトルは数を順に並べたもの、行列はベクトルを行または列に並べたものである。shape は各軸の長さを表す。たとえば token 数を \`T\`、隠れ次元を \`d\` と書けば、入力表現 \`X\` は通常 \`[T, d]\` と書ける。\n\n## 小さな例\n\n三つの token を使うなら \`T=3\` である。各 token が四つの数を持つ場合は \`X\` の shape は \`[3, 4]\` になる。大きな値を暗算する必要はない。まず「どの軸が token で、どの軸が特徴か」を声に出して確認する。\n\n## コードまたは擬似コード\n\n\`\`\`python\n# 入力の形を明示し、次の演算の出力を検査する\nassert len(x) == token_count\nnext_value = transform(x)\nprint("input", shape(x), "output", shape(next_value))\n\`\`\`\n\n## この比喩の限界\n\n直感は計算の入口であり、学習済み parameter の意味を人間の言葉で固定するものではない。特に attention の重みや embedding の一軸を、単独でモデルの思考や概念と同一視しない。\n\n## 実際の LLM との接続\n\n大規模な言語モデルでは同種の計算を、より多い層・head・次元・語彙で繰り返す。構成の詳細はモデルごとに異なるため、この章の例から全モデルの設計を断定しない。\n\n## 理解を確かめる\n\n${questions(title)}\n\n## 一文で言うと\n\n**${mechanism}**\n\n## 参照\n\n- [Vaswani et al., *Attention Is All You Need*](${source.url})\n`;
}

await mkdir("src/content/chapters", { recursive: true });
for (const chapter of chapters) {
  const [, order, title] = chapter;
  const slug = `${String(order).padStart(2, "0")}-${title.replace(/[,:・]/g, "").replaceAll(" ", "-").replaceAll("/", "-").toLowerCase()}`;
  await writeFile(`src/content/chapters/${slug}.md`, body(chapter), "utf8");
}

await writeFile("src/content/chapters/README.md", `# Content source\n\nThis directory contains 28 generated first-edition source chapters. Regenerate only after intentionally editing the source data in \`scripts/generate-content.mjs\`.\n`, "utf8");

const mathPages = [
  [1, "Vectors: 数を方向つきで並べる", "順番のある数の列としてのベクトル", "vector = [x₁, x₂, …, x_d]", "embedding、query、key、value は、同じ長さの数の列として計算される。"],
  [2, "Coordinates と dimensions", "座標と次元が何を数えているか", "shape([T, d]) = token axis × feature axis", "次元は人が個別に名付ける属性の一覧ではなく、モデルが学ぶ表現空間の座標数である。"],
  [3, "Dot product", "対応要素の積を足す比較", "a·b = Σ a_i b_i", "attention score は query と key の dot product から始まる。"],
  [4, "Matrix multiplication", "多数の線形変換をまとめて表す計算", "[T,d] × [d,h] → [T,h]", "Q/K/V の projection は、各 token に同じ重み行列を掛ける行列積で書ける。"],
  [5, "Transpose", "行と列を入れ替える操作", "(QKᵀ)[i,j] = Q[i]·K[j]", "key の行を列に替えると、全 query と全 key の組を一つの score matrix として計算できる。"],
  [6, "Functions", "入力を出力に対応させる規則", "f: input → output", "層は parameter を持つ関数であり、同じ形の入力に同じ手順を適用する。"],
  [7, "Exponents と logarithms", "softmax と loss に現れる二つの逆向きの演算", "softmax(z)_i = exp(z_i) / Σ exp(z_j)", "指数は score 差を強調し、log は正解確率が小さいときの損失を大きく表す。"],
  [8, "Probability distributions", "候補全体に割り振られた非負の重み", "p_i ≥ 0, Σ p_i = 1", "softmax 後の次 token 分布は、指定した語彙内での相対的な選択重みである。"],
  [9, "Derivatives と gradients", "微小な変化への反応を測る", "gradient = ∂loss / ∂parameter", "学習では parameter の各要素を少し動かしたとき、loss がどう変わるかをまとめて扱う。"],
  [10, "Chain rule", "つながった関数を後ろからたどる", "∂L/∂x = ∂L/∂y × ∂y/∂x", "backpropagation は、出力側の loss から入力側の parameter へ影響を連鎖させる手順である。"]
];

await mkdir("src/content/math", { recursive: true });
for (const [order, title, summary, formula, connection] of mathPages) {
  const slug = `${String(order).padStart(2, "0")}-${title.replaceAll(": ", "-").replaceAll(" ", "-").toLowerCase()}`;
  const content = `---\ntitle: ${yaml(title)}\norder: ${order}\nsummary: ${yaml(summary)}\nconcepts: [${yaml(title)}]\nreferences:\n  - label: ${yaml(source.label)}\n    url: ${yaml(source.url)}\n---\n\n# ${title}\n\n## これは何か\n\n${summary}。\`${formula}\` と書くと、Transformer の計算で現れる関係を短く表せる。\n\n## なぜここで必要か\n\n${connection}\n\n## 最小の計算\n\n数字を小さくして、各記号が何を数えるかを先に確認する。演算の結果よりも、入力と出力の shape がどう変わるかを追う。\n\n## コードとの接続\n\n\`\`\`python\n# 教材用: shape を先に表示してから演算する\nprint(shape(input_value))\nresult = operation(input_value)\nprint(shape(result))\n\`\`\`\n\n## 参照\n\n- [Vaswani et al., *Attention Is All You Need*](${source.url})\n`;
  await writeFile(`src/content/math/${slug}.md`, content, "utf8");
}

console.log(`Wrote ${chapters.length} chapters and ${mathPages.length} math pages across ${partNames.length} parts.`);
