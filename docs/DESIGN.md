# Transformer Atlas — Design

Status: working design v0.1

## 1. Purpose

Transformer Atlas is a Japanese interactive book/site that teaches Transformer architecture from conceptual intuition to a small working implementation.

The central educational promise is:

> Do not require the reader to become good at long calculations before allowing them to understand the architecture.

The site should explain what each mathematical operation is doing, why it appears, what information flows through it, and how it connects to the full model.

The goal is **architectural literacy**, not exam-style mathematical fluency.

## 2. Reader model

Assume a reader who:

- understands ordinary arithmetic and can follow middle-school algebra when explained;
- may have seen vectors, functions, exponents, probability, and matrices before but should not be assumed fluent in them;
- can understand a new mathematical idea if its purpose is clear;
- benefits from seeing the same concept through prose, diagram, tiny numeric example, and code;
- wants to know why a technique is used and where it fits in the system;
- does not need large sets of repetitive exercises.

Do not write for ML researchers by default. Do not infantilize the reader either.

## 3. Pedagogical rules

Every chapter should follow a recognizable rhythm where appropriate:

1. **The problem** — what problem must the model solve here?
2. **Intuition** — an accurate mental model in ordinary language.
3. **Mechanism** — what actually happens computationally.
4. **Math** — the smallest useful formal expression, with every symbol explained.
5. **Tiny example** — numbers small enough to inspect by eye.
6. **Interactive view** — where interaction materially improves understanding.
7. **Code** — a compact implementation or pseudocode mapping the concept to software.
8. **What this is not** — correct common oversimplifications introduced by the intuition.
9. **Why it matters in real LLMs** — connect the concept to modern systems.
10. **Check your understanding** — roughly 2–4 conceptual questions, preferably not arithmetic drills.
11. **In one sentence** — a concise chapter takeaway.

### Mathematics policy

- Explain a mathematical term at first use.
- Never use a formula merely to make the text look rigorous.
- Do not hide essential mechanics behind metaphor.
- Introduce notation only after the reader understands the problem it solves.
- Prefer tiny worked examples to large symbolic derivations.
- When a derivation matters, explain why each transformation is legitimate.
- Make dimensions/shapes visible whenever tensors or matrices are involved.
- Clearly distinguish scalar, vector, matrix, and tensor.
- Avoid assuming calculus, probability theory, or linear algebra as completed prerequisites.
- When calculus becomes relevant during training, teach only the conceptual minimum required to understand gradients and parameter updates, with optional deeper notes.

### Metaphor policy

Metaphors are allowed but must be explicitly returned to the real mechanism.

For example, Query / Key / Value can initially be described as a query, index, and retrieved information, but the chapter must then state that these are learned vector projections and show the actual computation.

## 4. Proposed curriculum

The exact chapter count may change during research, but the conceptual dependency order should remain coherent.

### Part I — From text to vectors

#### 1. What is a Transformer actually doing?

Start from the complete journey: text -> tokens -> vectors -> repeated Transformer blocks -> logits -> next-token distribution.

The reader should see the whole map before studying components.

#### 2. Tokens: text becomes pieces

- tokenization;
- token IDs;
- vocabulary;
- why tokens are not words;
- tokenization is not meaning extraction.

#### 3. Embeddings: IDs become vectors

- vectors as ordered lists / coordinates;
- dimensions;
- learned lookup table;
- why one number cannot conveniently represent all useful properties;
- similarity as geometry, with caveats.

#### 4. Dot products: comparing directions and relevance

- multiplication-and-sum intuition;
- geometric meaning where useful;
- why dot products are computationally convenient;
- relationship to attention scores.

### Part II — Attention

#### 5. The attention problem

Before Q/K/V, explain the core need: while processing one token, how should information from other positions be weighted?

Use an interactive sentence where attention weights can be inspected.

#### 6. Query, Key, Value

- learned linear projections;
- `Q = XW_Q`, `K = XW_K`, `V = XW_V`;
- shapes;
- scores from `QK^T`;
- why the metaphor is useful and where it breaks.

#### 7. Scaling and softmax

- why raw dot products need scaling;
- `sqrt(d_k)` conceptually;
- exponentials and normalization;
- softmax as relative weighting, not certainty;
- interactive score -> probability sliders.

#### 8. Self-attention end to end

Build the full equation carefully:

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`

The reader should be able to point to every term and explain its job.

#### 9. Multi-head attention

- multiple learned projection spaces;
- concatenate and project;
- avoid claiming that individual heads always correspond cleanly to human categories such as grammar or sentiment;
- interactive simplified multi-head view.

### Part III — A Transformer block

#### 10. Position: order must enter somehow

Cover sequence order and the reason content-only attention is insufficient.

Introduce sinusoidal position encoding historically, then explain modern rotary positional embeddings (RoPE) conceptually.

#### 11. Feed-forward networks

- per-position transformation;
- expansion / contraction;
- nonlinear activation;
- why attention is not the whole Transformer.

#### 12. Residual connections and normalization

- preserving information paths;
- optimization stability intuition;
- LayerNorm/RMSNorm at an appropriate level;
- pre-norm vs post-norm only as an optional note.

#### 13. One complete Transformer block

Interactive block diagram showing data flow through attention, residual path, normalization, and MLP.

### Part IV — From Transformer to GPT

#### 14. Encoder, decoder, and decoder-only models

Explain the original Transformer architecture, then why GPT-style models are decoder-only.

#### 15. Causal masking

- no looking ahead during next-token prediction;
- triangular mask;
- interactive visible/hidden attention matrix.

#### 16. Logits and next-token prediction

- output projection;
- logits;
- softmax distribution;
- sampling vs argmax.

#### 17. Temperature, top-k, and top-p

Show generation controls as transformations of a probability distribution, not personality knobs.

### Part V — Learning

#### 18. What exactly gets learned?

- parameters;
- weights;
- embeddings and projection matrices;
- initialization;
- the difference between architecture and learned contents.

#### 19. Loss: how the model knows it was wrong

- target token;
- cross-entropy conceptually;
- negative log probability without requiring advanced probability background.

#### 20. Gradients and backpropagation without turning this into a calculus course

- local sensitivity;
- chain rule intuition;
- gradient as direction for parameter adjustment;
- optional mathematical appendix.

#### 21. Training loop

A tiny training-loop animation: batch -> forward -> loss -> backward -> optimizer -> updated parameters.

### Part VI — Running modern LLMs

#### 22. Context windows

- sequence length;
- attention cost;
- what context means operationally.

#### 23. KV cache

Explain why autoregressive generation can reuse previous K/V tensors and what tradeoffs result.

#### 24. Modern variations

A survey chapter clearly labeled as beyond the core Transformer:

- RoPE;
- RMSNorm;
- SwiGLU / gated MLPs;
- grouped-query / multi-query attention;
- Mixture of Experts;
- long-context techniques.

Do not imply every modern LLM uses every technique.

#### 25. What the Transformer does not explain by itself

Separate core architecture from:

- data curation;
- tokenizers;
- alignment / post-training;
- tool use;
- RAG;
- inference-time reasoning strategies;
- system prompts;
- agent orchestration.

### Part VII — Build one

#### 26. Build a tiny Transformer from scratch

Implement a deliberately tiny decoder-only language model in understandable stages.

Every significant code block should link back to the relevant conceptual chapter.

Possible implementation choices: Python + PyTorch for readability, while ensuring that framework calls do not hide the mechanism being taught.

#### 27. Trace one token through the model

Use a concrete miniature model and inspect shapes/data flow from token ID to logits.

#### 28. Final map

Return to the full architecture diagram from Chapter 1. Everything should now be clickable and understood.

## 5. Optional math side-paths

Do not force all readers through prerequisite chapters before Transformer content.

Provide context-sensitive side pages such as:

- vectors;
- coordinates and dimensions;
- dot product;
- matrix multiplication;
- transpose;
- functions;
- exponents and logarithms;
- probability distributions;
- derivatives / gradients;
- chain rule.

Every main chapter may link to these pages when needed.

Each math page should answer three questions:

1. What is this idea?
2. Why does Transformer need it here?
3. What is the minimum calculation I need to follow the mechanism?

## 6. Interactive visualizations

Interactive components should teach a specific concept. Avoid decorative animation.

Priority components:

### Tokenizer explorer

Enter a short sentence and inspect token boundaries and IDs using a fixed demonstrative tokenizer or safely bundled example data. Avoid requiring a hosted API.

### Embedding explorer

Show a small synthetic embedding space. Let the reader move points and observe similarity / dot product changes.

Do not imply that a 2D projection faithfully represents a production embedding space.

### Dot-product lab

Manipulate two small vectors and see the elementwise products, sum, angle intuition, and resulting score.

### Attention sentence explorer

Click a token and visualize its attention weights to other tokens. Use explicitly illustrative or locally computed small-model data and label it accordingly.

### Q/K/V pipeline

Step through projection -> score matrix -> scaling -> softmax -> weighted sum.

### Softmax sandbox

Sliders alter input scores and immediately show normalized weights. Include temperature later as a comparison.

### Causal-mask matrix

Toggle mask visibility and show which positions are inaccessible.

### Transformer block flow

Animate one vector through attention / residual / norm / MLP paths while retaining tensor-shape labels.

### Generation sampler

Given a fixed list of logits, change temperature/top-k/top-p and sample repeatedly.

### KV-cache explorer

Compare recomputing all prior keys/values with cache reuse.

### Tiny-model trace

Display a small, deterministic forward pass and connect each step to code and equations.

## 7. Accuracy requirements

This is educational material about a technical subject, so factual review is part of implementation.

For claims about the original Transformer, prioritize the original paper.

For PyTorch/framework behavior, use official documentation.

For architecture variants such as RoPE, RMSNorm, FlashAttention, GQA, MoE, etc., use the relevant primary papers or official project documentation.

Avoid relying on unsourced blog simplifications for technical claims when primary sources are available.

Keep a source/provenance file for each chapter or a structured references section.

Distinguish clearly among:

- facts about the original 2017 Transformer;
- common modern practices;
- implementation-specific choices;
- pedagogical simplifications;
- speculative interpretation.

## 8. Content format

Prefer one MD or MDX source file per chapter.

Suggested layout:

```text
src/content/
  chapters/
    01-transformer-map.mdx
    02-tokens.mdx
    ...
  math/
    vectors.mdx
    dot-product.mdx
    matrices.mdx
    ...
```

Chapter frontmatter should support at least:

```yaml
title:
part:
order:
summary:
prerequisites:
concepts:
interactive_components:
references:
status:
```

Keep content separable from presentation so the book can later be reused for print/PDF/EPUB if desired.

## 9. Site architecture

Preferred initial stack:

- Astro;
- MDX for chapter content requiring components;
- static generation;
- TypeScript;
- minimal client-side islands for interaction;
- KaTeX or another appropriate static math renderer;
- no database;
- no account system;
- no mandatory external API.

Prioritize static, durable content and only hydrate interactive sections.

Avoid turning the site into a SPA merely because interactions exist.

## 10. UX

### Navigation

Provide:

- ordered curriculum navigation;
- previous / next chapter;
- part overview;
- concept index;
- math reference index;
- full Transformer map;
- search if straightforward and static-friendly.

### Reading state

Do not require accounts. Local progress tracking may be added with localStorage if it improves usability and remains optional.

### Responsive design

The prose must read comfortably on phones and desktop. Dense matrices/diagrams may use horizontally scrollable or responsive layouts.

### Accessibility

- semantic HTML;
- keyboard-operable interactions;
- visible focus states;
- reduced-motion support;
- explanatory text alternatives for diagrams;
- do not encode meaning by color alone;
- math must remain understandable with assistive technology as far as practical.

### Visual character

Aim for a clean technical atlas / annotated-diagram feel rather than generic AI neon decoration.

The visual system should support diagrams, matrices, code, formulas, and explanatory prose as first-class content.

## 11. Assessment design

Questions should test mental models more than arithmetic speed.

Good examples:

- Why can attention not use token IDs directly as meaningful vectors?
- What changes when Q changes but K and V remain fixed?
- Why is causal masking required during autoregressive training?
- Which part of a Transformer mixes information across positions, and which transforms positions independently?

Where useful, allow interactive answer reveal/explanation without server storage.

Avoid gamification that distracts from understanding.

## 12. Code policy

Code examples should be executable where feasible and intentionally small.

For the final tiny model:

- avoid helper abstractions that hide attention internals;
- label tensor dimensions;
- keep random seeds fixed for reproducibility;
- keep model/data tiny enough for ordinary local execution;
- do not require GPU hardware;
- provide expected output where useful;
- separate educational implementation from production optimization.

## 13. Research and copyright boundary

Do not copy textbook prose, diagrams, course materials, or copyrighted explanatory content.

Use primary sources for facts, then write original explanations and original diagrams.

Short quotations should only be used when genuinely necessary and properly attributed.

The repository should maintain clear reference metadata so technical assertions can be audited.

## 14. Quality gates

Before treating a chapter as complete, check:

- conceptual dependency: no unexplained prerequisite appears;
- mathematical symbols are defined;
- metaphor returns to mechanism;
- code agrees with prose;
- diagram agrees with code/math;
- claims have credible sources;
- no statement implies an illustrative visualization is actual hidden model behavior unless it is;
- comprehension questions are answerable from the chapter;
- accessibility and mobile layout are acceptable.

## 15. Build phases

Manus may modify these phases if research shows a better order, but should preserve the end goal.

### Phase A — research and curriculum audit

- verify curriculum against primary sources and strong educational references;
- identify common learner misconceptions;
- decide exact chapter dependency graph;
- define source policy and glossary.

### Phase B — site system

- Astro/MDX scaffold;
- content collections/schema;
- typography/math/code system;
- navigation;
- component framework;
- tests/CI.

### Phase C — core book

Write the complete core curriculum, not only sample chapters.

Prioritize Parts I–V and the tiny model path before optional modern-architecture depth.

### Phase D — interactive atlas

Implement the high-value visualizations that materially improve understanding.

### Phase E — tiny Transformer

Build and document the reproducible miniature model and trace.

### Phase F — review and polish

- technical fact-check;
- cross-link concepts;
- accessibility;
- responsive QA;
- content consistency;
- build/test/CI;
- deployment documentation.

## 16. Definition of success

A successful first release allows a motivated non-specialist reader to:

1. describe the overall text-to-next-token pipeline;
2. explain tokens and embeddings;
3. explain dot-product attention and every term in the attention equation;
4. explain Q/K/V as actual learned projections rather than only metaphor;
5. explain multi-head attention, positional information, MLPs, residual paths, and normalization;
6. distinguish the original Transformer from GPT-style decoder-only models;
7. explain causal masking and token generation;
8. understand at a high level what training updates and why;
9. explain context windows and KV caching;
10. read a small Transformer implementation and connect its lines back to the conceptual chapters.

The reader should finish feeling that the architecture is inspectable machinery, not magic.
