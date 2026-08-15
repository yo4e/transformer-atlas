import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";

const modelForCore = "gpt-5-mini";
const modelForGeneral = "gpt-5-mini";
const coreOrders = new Set([3, 4, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22, 23, 26, 27]);
const sources = `
- [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Sennrich et al. (2016), Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Radford et al. (2019), Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Su et al. (2021), RoFormer](https://arxiv.org/abs/2104.09864)
- [Zhang & Sennrich (2019), RMSNorm](https://arxiv.org/abs/1910.07467)
- [Ainslie et al. (2023), GQA](https://arxiv.org/abs/2305.13245)
- [Shazeer et al. (2017), Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)
- [Dao et al. (2023), FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [PyTorch: Embedding / CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)`;

const chapters = [
  [1, "01-transformer-は何をしているのか.md", "Transformer は何をしているのか", "文字列が token ID、embedding、複数の block、logits、次 token 分布になる全体地図。architecture と parameter を区別し、各矢印の前後の情報を言葉で追う。", "語彙 {猫:2, は:5, 寝る:9, EOS:0} と token 列 [2,5,9] を、[3]→[3,4]→[3,4]→[3,10] の表として追う。"],
  [2, "02-tokens-文字列が部品になる.md", "Tokens: 文字列が部品になる", "文字列を有限語彙の token ID に変える tokenizer。日本語・英語・未知語の分割は tokenizer に依存し、token は単語ではない。", "『青い猫』と 'unhappiness' を、教育用の subword 片と ID の列にし、別の tokenizer なら別の列になることを示す。"],
  [3, "03-embeddings-id-がベクトルになる.md", "Embeddings: ID がベクトルになる", "ID は計算可能な意味を持たない。embedding table の一行を引く lookup と、学習で表の値が変わる意味を説明する。", "語彙4、次元3の embedding table を置き、ID [2,0,3] がどの三行を選ぶかを実際に表にして追う。"],
  [4, "04-dot-product-向きと関連度を比べる.md", "Dot product: 向きと関連度を比べる", "同じ長さの二つの vector を一つの score にする dot product。符号と大きさが score に与える影響を attention の比較へつなぐ。", "a=[2,-1,1] と b=[3,2,-2]、c=[2,-1,0] を計算し、a·b と a·c の違いを各積の和として示す。"],
  [5, "05-attention-が解く問題.md", "Attention が解く問題", "各位置が入力ごとに異なる位置から情報を集める必要。固定窓や一律平均との差を、重み付き和として説明する。", "三つの value [2,0]、[0,3]、[1,1] に重み [0.6,0.3,0.1] を掛けて出力を計算し、一律平均と比較する。"],
  [6, "06-query-key-value.md", "Query, Key, Value", "同じ入力 X から三種類の学習済み線形射影を作る理由。Q は比較の観点、K は照合に使う特徴、V は混合される内容であり、擬人化の限界も説明する。", "X=[[1,0],[0,1]]、W_Q=[[1,1],[0,1]]、W_K=[[1,0],[1,-1]]、W_V=[[2,0],[1,1]] から Q,K,V を全行列積で計算する。"],
  [7, "07-scaling-と-softmax.md", "Scaling と softmax", "任意実数の score を、value を混ぜる正の比率に変える必要。負の値、尺度、順序保存、指数と正規化、temperature を説明する。", "scores=[2,1,-1] を exp の近似値 [7.39,2.72,0.37] にして合計10.48で割り、temperature 0.5 と2の違いも比較する。"],
  [8, "08-self-attention-を最後まで通す.md", "Self-attention を最後まで通す", "QK^T、scale、softmax、V の重み付き和までを一つの入力から隠れた工程なしに通す。各行・列と各行列積の役割を分ける。", "Q=[[1,0],[0,1]]、K=[[1,1],[1,-1]]、V=[[2,0],[0,3]] を用い、score、scale、各行softmax、output の数値を順番にすべて計算する。"],
  [9, "09-multi-head-attention.md", "Multi-head attention", "複数 head は入力を異なる低次元射影で並列に計算し、結合後に混ぜる。head に固定した人間の意味ラベルを与えない。", "d_model=4、head数2、d_head=2 の [2,4] 入力を二つの [2,2] head に射影し、concat が [2,4] に戻る流れを示す。"],
  [10, "10-position-順序を入れる.md", "Position: 順序を入れる", "attention 単体は並び替えに鈍感なので、位置を渡す必要がある。原著の加算型 sinusoidal encoding と後年の RoPE を厳密に分ける。", "二つの token embedding [1,0] に位置 vector [0,1] と [1,0] を足す例、および2次元 vector [1,0] を90度回す RoPE の最小例を用いる。"],
  [11, "11-feed-forward-network.md", "Feed-forward network", "attention で位置間を混ぜた後、各位置に同じ非線形関数を独立適用する MLP。位置間混合と特徴変換の役割を分ける。", "x=[1,-1]、W1=[[1,2],[0,1]]、ReLU、W2 の小さな数値例を使い、ReLU 前後と出力を追う。"],
  [12, "12-residual-connection-と-normalization.md", "Residual connection と normalization", "深い層で入力を保つ残差経路と、値の規模を扱いやすくする LayerNorm/RMSNorm。original Transformer と近年の選択を分ける。", "x=[2,-1] と sublayer(x)=[0.5,1] の足し戻し、さらに [2,4] の平均・分散を使う教育用 LayerNorm を計算する。"],
  [13, "13-一つの-transformer-block.md", "一つの Transformer block", "norm、attention、残差、MLP、残差を順番に合成した block。pre-norm と原著の構成は混同せず、教育用 block の情報経路を追う。", "二 token の [2,2] 表現に、既出の小 attention 出力と小 MLP 出力を足し戻す表を作る。"],
  [14, "14-encoder-decoder-decoder-only.md", "Encoder, decoder, decoder-only", "2017年原著の encoder-decoder と、因果マスク付き decoder-only GPT 型の目的・情報経路の違い。", "入力『猫が寝る』を翻訳する encoder-decoder の注意範囲と、prefix『猫が』から次 token を予測する decoder-only の範囲を行列表で比較する。"],
  [15, "15-causal-masking.md", "Causal masking", "次 token を当てる学習で未来を先読みしないため、softmax 前の score に加える上三角 mask。token を消すのではない。", "長さ3の score 行列に -∞ を入れ、softmax 後に [1,0,0]、[*,*,0] となる具体的な三行を示す。"],
  [16, "16-logits-と次-token-予測.md", "Logits と次 token 予測", "最後の hidden vector を語彙ごとの未正規化 score にする output projection。logit、確率、選択結果は別物。", "h=[1,2]、語彙4、W_vocab の小表から logits [2,1,3,-1]、softmax 確率、argmax を計算する。"],
  [17, "17-temperature-top-k-top-p.md", "Temperature, top-k, top-p", "同じ logits 分布から token を選ぶ複数の規則。学習時の softmax と生成時の sampling 方針を区別する。", "確率 [0.55,0.25,0.12,0.08] に temperature、top-k=2、top-p=0.8 を順に適用し、再正規化の数値を示す。"],
  [18, "18-何が学習されるのか.md", "何が学習されるのか", "architecture は計算の型、parameter は更新される数値。embedding、投影行列、norm の scale 等が同じ入力に対する活性化を変える。", "2×2の W_Q の一要素を 0.3 から0.4へ変えたとき、同じ x=[2,1] の投影がどう変わるかを計算する。"],
  [19, "19-loss-間違いを数値にする.md", "Loss: 間違いを数値にする", "次 token の正解確率を大きくする学習信号としての negative log likelihood / cross-entropy。確率が低いほど損失が大きい理由を示す。", "正解 token の確率0.8と0.2について -log の近似値0.22と1.61を比較し、同じ target に対する差を説明する。"],
  [20, "20-gradient-と-backpropagation.md", "Gradient と backpropagation", "parameter を少し動かすと loss がどちらへ変わるかという局所感度。微積の前提を最小にし、連鎖の考えを数値で説明する。", "L(θ)=(θ-3)^2、θ=1での傾き-4、学習率0.1の更新 θ=1.4 を計算し、ニューラルネットの多パラメータ版へつなぐ。"],
  [21, "21-training-loop.md", "Training loop", "batch、forward、logits、loss、backward、optimizer update の反復。訓練時の重み更新と推論時の活性化計算を明確に分離する。", "prefix [2,5]→target9 の一例に、logits、target loss、W の一要素更新を小さな擬似数値で通す。"],
  [22, "22-context-window.md", "Context window", "一回の forward pass に渡す token 範囲と attention score 行列の長さ二乗。会話の意味としての文脈だけではない。", "T=4 とT=8で score 要素が16と64になる表、long context の実装上の工夫を後年の選択として示す。"],
  [23, "23-kv-cache.md", "KV cache", "自己回帰生成で過去 K/V が変わらないための再利用。訓練時の全列並列計算と生成時の1 tokenずつ計算を区別する。", "t=1,2,3の K,V cache を [t,d_k]、[t,d_v] として表にし、新 query が過去三行へ score を作る例を示す。"],
  [24, "24-modern-variations.md", "Modern variations", "RoPE、RMSNorm、SwiGLU、GQA、FlashAttention、MoE は別々の問題に対する後年の変種・実装技術であり、全LLM共通ではない。", "d_model=8、query head4、KV head2 のGQAを小表で示し、RoPE・RMSNorm・MoEを目的/場所/注意点で比較する。"],
  [25, "25-transformer-だけでは説明できないこと.md", "Transformer だけでは説明できないこと", "architecture、training data、objective、post-training、retrieval、tools、system design を同一視しない。能力と内部機構の因果を単純化しない。", "同じ小 architecture に異なる token 列を学習させる思考実験と、model weights・retrieved document・tool result の三層表を用いる。"],
  [26, "26-小さな-transformer-を組み立てる.md", "小さな Transformer を組み立てる", "examples/tiny_transformer.py を、embedding、QKV、score、mask、softmax、mixing、residual、MLP、logits の章へ逆リンクできる実行可能な伴走者にする。", "実装にある小 vocab、token_ids、d_model、mask の実値を使い、各関数の入出力を章番号つき表にする。"],
  [27, "27-一つの-token-を追跡する.md", "一つの token を追跡する", "一つの位置の token ID が embedding、layer内のQKV、attention output、hidden、logitsを経る trace。各段階は別の種類の数値である。", "位置1の token ID=2を仮定し、[T]→[T,d]→score row[T]→output[d]→logits[vocab] の具体表を埋める。"],
  [28, "28-final-map.md", "Final map", "全体地図に戻り、text→IDs→vectors→blocks→logits→selection の各矢印に問題、演算、shape、コード箇所を対応づける。", "三 token の小例を最初から最後まで要約表にし、読者が説明できる問いと次の学習経路を示す。"]
];

function splitFrontmatter(text) {
  const end = text.indexOf("\n---", 4);
  if (end < 0) throw new Error("frontmatter missing");
  return text.slice(0, end + 4);
}

function cleanDraft(text) {
  return text.replace(/^```(?:markdown)?\s*/i, "").replace(/\s*```$/i, "").replace(/^#\s+.*\n+/m, "").trim();
}

function promptFor(order, title, focus, example) {
  const depth = coreOrders.has(order) ? "7,000〜10,000文字" : "5,000〜7,000文字";
  return `あなたは、大学数学を前提にしない成人向けの日本語技術教科書『Transformer Atlas』の筆者です。第${order}章「${title}」を、既存の薄いテンプレートから独立して読める実用的な第二稿へ書き直してください。

読者は四則演算と中学程度の代数を読めますが、線形代数・確率・微積・深層学習の履修を前提にしてはいけません。知的水準は下げず、前提負担だけを下げてください。出力は frontmatter と # 見出しを含めない Markdown 本文だけにしてください。日本語の実質的な説明量は目安として${depth}です。箇条書きの羅列や同じ定型句の反復で水増しせず、自然な段落を中心に書いてください。

この章の焦点: ${focus}
必ず本文に通す具体例: ${example}

構成要件:
1. 問題を先に提示する。なぜこの仕組みが必要かを、仕組みの名前を答えにせず説明する。
2. 有用な直感を示し、直感が文字どおりではない点を明示する。
3. 実際の数値オブジェクト・学習parameter・入力から計算されるactivationを区別する。
4. 上の具体例を省略せず、数を実際に計算する。shapeだけで説明を終えない。
5. 動機の後に数式を示し、全ての記号・行列積・軸の意味を普通の言葉に戻す。
6. 透明な短いPythonコードを示す。変数名は数式と対応させ、transform(x)のような仮コードは禁止する。
7. full Transformer 内での前後の処理、現実の大規模モデルとの接続、少なくとも一つのもっともらしい誤解を訂正する。
8. 章の最後に、読者が具体的に何を説明できるか、確認問題2問、関連資料のMarkdownリンクを置く。

正確性の規則: original Transformer、GPT型decoder-only、後年の変種を混同しない。教材用の小数値は実モデルの内部状態ではないと明記する。tokenは単語と同義ではない。attention weightを人間的な『注目』の完全な証拠として扱わない。modern variationsを扱う場合は採用状況を一般化しない。

参照候補（章に関係するものだけを本文末に引用）:
${sources}
`;
}

async function invoke(model, prompt) {
  const response = await fetch(`${process.env.OPENAI_API_BASE}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: "You write accurate, original Japanese technical textbooks. Return only the requested Markdown." },
        { role: "user", content: prompt }
      ],
      max_completion_tokens: 5000,
      reasoning: { effort: "minimal" }
    })
  });
  if (!response.ok) throw new Error(`LLM ${response.status}: ${await response.text()}`);
  const payload = await response.json();
  return payload.choices?.[0]?.message?.content ?? "";
}

async function writeOne(item) {
  const [order, filename, title, focus, example] = item;
  const target = path.join("src/content/chapters", filename);
  const original = await readFile(target, "utf8");
  const frontmatter = splitFrontmatter(original);
  const model = coreOrders.has(order) ? modelForCore : modelForGeneral;
  let draft = "";
  let lastError = "";
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      draft = cleanDraft(await invoke(model, `${promptFor(order, title, focus, example)}\nこれは${attempt}回目の執筆です。短い概説ではなく、必要な数値経路と説明を完結させてください。`));
      if (draft.length >= 4200) break;
      lastError = `draft too short: ${draft.length}`;
    } catch (error) {
      lastError = String(error);
    }
  }
  if (draft.length < 4200) throw new Error(`${title}: ${lastError}`);
  await writeFile(target, `${frontmatter}\n\n# ${title}\n\n${draft}\n`, "utf8");
  return { order, title, characters: draft.length, model };
}

async function runPool(items, workers = 4) {
  const results = [];
  let cursor = 0;
  async function worker() {
    while (cursor < items.length) {
      const current = items[cursor++];
      const result = await writeOne(current);
      results.push(result);
      console.log(`${String(result.order).padStart(2, "0")} ${result.title}: ${result.characters} chars via ${result.model}`);
    }
  }
  await Promise.all(Array.from({ length: workers }, worker));
  return results.sort((a, b) => a.order - b.order);
}

await mkdir("docs/research", { recursive: true });
const results = await runPool(chapters);
const total = results.reduce((sum, result) => sum + result.characters, 0);
await writeFile("docs/research/SECOND_DRAFT_GENERATION.json", JSON.stringify({ generatedAt: new Date().toISOString(), totalCharacters: total, chapters: results }, null, 2) + "\n", "utf8");
console.log(`Second draft complete: ${total} body characters across ${results.length} chapters.`);
