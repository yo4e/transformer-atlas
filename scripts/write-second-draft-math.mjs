import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const notes = [
  ["01-vectors-数を方向つきで並べる.md", "Vectors: 数を方向つきで並べる", "vectorは順序を持つ数の列であり、embedding・Q・K・Vの最小単位。座標に人間用の意味ラベルを過剰に与えない。", "[2,-1,3] と [1,4,-2] を使い、要素ごとの加算、定数倍、長さが同じである必要を計算する。"],
  ["02-coordinates-と-dimensions.md", "Coordinates と dimensions", "axis、coordinate、dimension、shapeの違い。token axisとfeature axisを混同せず、[T,d]が何を数えるかを理解する。", "三token・二featureの [[1,0],[2,1],[-1,3]] を表にし、行と列を入れ替えたときに何が変わるかを示す。"],
  ["03-dot-product.md", "Dot product", "dot productは二vectorの対応要素の積の和であり、attention scoreの素材。cosine similarityと同一視しない。", "[2,-1,1]・[3,2,-2] と [2,-1,0] の計算を逐一示し、正負と大きさの寄与を比較する。"],
  ["04-matrix-multiplication.md", "Matrix multiplication", "行列積を『行と列のdot productを一度に並べる』操作として、QKV projectionと結び付ける。", "X=[[1,0],[0,1]] と W=[[1,2],[3,4]] から XW を全四要素計算し、[2,2]×[2,2]の軸を追う。"],
  ["05-transpose.md", "Transpose", "transposeは行列の行と列の役割を交換する操作。K^Tにより全queryと全keyの比較が並ぶ理由を説明する。", "K=[[1,1],[1,-1],[0,2]] を転置し、Q=[ [1,0],[0,1] ] との QK^T の全要素を計算する。"],
  ["06-functions.md", "Functions", "層は入力を出力へ対応させるparameter付き関数。訓練時にparameterが更新され、推論時は固定される。", "f(x)=2x+1、ReLU、xWの三例で、同じ入力が関数によって別表現へ写ることを示す。"],
  ["07-exponents-と-logarithms.md", "Exponents と logarithms", "softmaxのexpとnegative log likelihoodのlogを、互いに逆の役割を持つ演算として最小限に導入する。", "scores [0,1,2] の exp近似と正規化、確率0.8と0.2の -log近似を計算する。"],
  ["08-probability-distributions.md", "Probability distributions", "非負で合計1の重み列としての分布。attentionの重みと次token確率は用途が異なるが、同じ正規化の性質を持つ。", "[0.5,0.3,0.2] と重み付きvalueの例、語彙三個の次token分布を比較する。"],
  ["09-derivatives-と-gradients.md", "Derivatives と gradients", "gradientはparameterの微小変化に対するlossの局所的な感度。高次元での『矢印』の比喩の限界も説明する。", "L(θ)=(θ-3)^2の θ=1 とθ=4の傾き、二parameterの小さな更新を計算する。"],
  ["10-chain-rule.md", "Chain rule", "連結した関数への影響を途中の変化率の積として戻す考え方。backpropagationは多数parameter版の効率的計算。", "z=2θ、L=(z-4)^2で dL/dz と dz/dθ を計算し、dL/dθを求める。"]
];

function frontmatter(text) { const end = text.indexOf("\n---", 4); if (end < 0) throw new Error("frontmatter missing"); return text.slice(0, end + 4); }
function clean(text) { return text.replace(/^```(?:markdown)?\s*/i, "").replace(/\s*```$/i, "").replace(/^#\s+.*\n+/m, "").trim(); }
function prompt(title, topic, example) { return `あなたは大学数学の前提なしにTransformerを読み解く日本語教材の筆者です。数学ノート「${title}」を独立して役に立つ第二稿へ書いてください。本文のみをMarkdownで返し、frontmatterと#見出しは含めません。3,500〜5,500文字を目安に、まとまった段落で書いてください。

主題: ${topic}
必ず通す数値例: ${example}

必要な構成: なぜTransformerのどの箇所で必要になるか、最小の定義、上の数値を省略せず計算する小見出し、記号とshapeの解読、透明な短いPythonコード、よくある誤解、本文の関連章へのリンク、読者が説明できること、参照。手計算の速さを要求せず、数が何を表すかを説明してください。仮コードは禁止です。原著のTransformerに関する事実はVaswani et al. 2017、後年の変種は明示的に区別してください。`; }
async function call(promptText) { const response = await fetch(`${process.env.OPENAI_API_BASE}/chat/completions`, { method:"POST", headers:{"Content-Type":"application/json",Authorization:`Bearer ${process.env.OPENAI_API_KEY}`}, body:JSON.stringify({model:"gpt-5-mini",messages:[{role:"system",content:"Write accurate, original Japanese teaching material. Return only Markdown."},{role:"user",content:promptText}],max_completion_tokens:4000,reasoning:{effort:"minimal"}}) }); if (!response.ok) throw new Error(await response.text()); const data=await response.json(); return data.choices?.[0]?.message?.content ?? ""; }
async function one(note) { const [filename,title,topic,example]=note; const target=path.join("src/content/math",filename); const front=frontmatter(await readFile(target,"utf8")); let body=""; for(let attempt=0;attempt<3 && body.length<3000;attempt+=1) body=clean(await call(`${prompt(title,topic,example)}\n第${attempt+1}稿。`)); if(body.length<3000) throw new Error(`${title}: too short`); await writeFile(target,`${front}\n\n# ${title}\n\n${body}\n`,"utf8"); console.log(`${title}: ${body.length}`); return body.length; }
let index=0; const sizes=[]; await Promise.all(Array.from({length:3},async()=>{while(index<notes.length){sizes.push(await one(notes[index++]));}})); console.log(`Math second draft complete: ${sizes.reduce((a,b)=>a+b,0)} characters.`);
