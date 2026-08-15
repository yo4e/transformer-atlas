# Manus Brief — Second Draft: Write the Actual Book

## Mission

The current Transformer Atlas is a useful site skeleton and curriculum outline, but its chapter prose is still only a first-pass scaffold. Treat every existing chapter as **an outline that must be rewritten into a real textbook chapter**.

The next task is not to add more infrastructure, more checkboxes, or more superficial features. The task is to **write the book**.

Preserve the existing Astro site, navigation, content collection, tests, examples, and interactive framework unless a change is genuinely required to support the rewritten content. Spend the overwhelming majority of effort on explanation quality, worked examples, mathematical meaning, diagrams/interactions that directly teach a concept, and technical accuracy.

Do not stop after revising a few representative chapters. Continue through the full curriculum in one autonomous pass as far as context and execution limits allow.

---

## 1. Reader model

Write for an intelligent adult reader who:

- can follow ordinary arithmetic and middle-school algebra;
- may remember vectors and functions only vaguely;
- has not studied university linear algebra, probability, calculus, or deep learning formally;
- can read small Python examples when every line has a purpose;
- does **not** want to become fast at hand calculation;
- instead wants to understand what mathematical object is being used, why it is useful, what information it carries, and where it sits in the Transformer;
- ultimately wants to look at an implementation and recognize what each major operation is doing.

Do not lower the intellectual level. Lower the **prerequisite burden**.

The reader should be allowed to understand a matrix multiplication before becoming skilled at manually multiplying large matrices.

---

## 2. Current content is not "complete"

Existing chapter frontmatter may say `status: complete`. Do not treat that as evidence that the chapter is finished.

The current prose is often only a few kilobytes and commonly follows this pattern:

- short problem statement;
- one-paragraph intuition;
- one formula;
- generic `[T, d]` shape example;
- generic pseudocode;
- comprehension questions.

That is a **chapter scaffold**, not the finished learning material.

Rewrite rather than merely append padding.

Avoid mechanical expansion such as repeating the same explanation in different words. Every added section must answer a real reader question or advance the mechanism.

---

## 3. Target depth and length

Use length as a guardrail against under-explanation, not as a quota to fill with repetition.

General chapters should usually reach roughly **5,000–9,000 Japanese characters of substantive prose**, excluding frontmatter and references.

Core mechanism chapters may naturally reach **8,000–15,000+ Japanese characters** when needed, especially:

- Embeddings
- vectors and dot products
- Query / Key / Value
- scaled dot-product attention
- Softmax
- self-attention end-to-end
- multi-head attention
- positional information / RoPE
- residual connections and normalization
- feed-forward networks
- causal masking
- logits and next-token prediction
- training / loss / backpropagation at the conceptual level
- inference, KV cache, sampling

A coherent full course/book of approximately **100,000–150,000 Japanese characters or more** is acceptable if the material earns the length.

Do not truncate a concept simply to keep chapters uniform.

---

## 4. Required explanatory arc for every important concept

For every major mechanism, explicitly cover the following sequence where applicable.

### A. The problem before the mechanism

Explain what problem exists before introducing the solution.

Bad:

> Softmax converts scores to weights.

Better structure:

1. attention has produced arbitrary real-valued compatibility scores;
2. we need numbers that can sensibly control how much information to mix;
3. simply dividing by the sum has problems with negative values and scale;
4. Softmax provides a positive normalized weighting while preserving ordering;
5. then show what it actually computes.

### B. Plain-language intuition

Give a useful mental model, but clearly mark where the analogy stops being literal.

### C. The actual object

State what the thing really is.

Examples:

- an embedding is a learned vector lookup, not a dictionary definition;
- Q/K/V are learned linear projections, not literal questions, keys, and stored documents;
- an attention weight is a numerical mixing coefficient, not proof that the model "looked at" something in a human cognitive sense.

### D. A tiny numeric example

Use numbers small enough that a reader can follow every operation.

Do not merely show shapes. Actually compute representative values.

For example, for attention use perhaps 2–4 tokens and 2–3 dimensions and show:

1. `X`;
2. `W_Q`, `W_K`, `W_V`;
3. `Q`, `K`, `V`;
4. `QK^T`;
5. scaling;
6. masking if relevant;
7. Softmax;
8. weighted sum of `V`;
9. resulting output vectors.

A reader should be able to trace where every number came from.

### E. The formula

Only after motivation and a concrete example, show the compact formula.

Then decode the formula symbol by symbol in ordinary language.

For example, do not stop at:

`Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V`

Explain what each multiplication changes, what each axis means, and why the final multiplication by `V` is conceptually different from the `QK^T` comparison.

### F. Shape tracking

Track tensor/matrix shapes through the mechanism.

Do not repeatedly use shape notation as a substitute for mechanism explanation.

Explain why the dimensions must line up and what each axis represents.

### G. Code that corresponds to the explanation

Code must implement the exact concept just explained.

Avoid placeholder pseudocode such as:

```python
next_value = transform(x)
```

unless the point of the section is specifically abstraction.

Prefer small transparent code whose variables match the notation used in the chapter.

### H. Connection to the full Transformer

Show where this operation sits in the block/model and what comes immediately before and after it.

### I. Common misconception

Name at least one plausible misunderstanding and correct it.

### J. What the reader can now explain

End with a short competence statement more specific than generic quiz questions.

Example:

> You should now be able to explain why Q and K determine *how much* to mix while V determines *what content* gets mixed.

---

## 5. Stop reusing generic examples

The existing first draft often repeats variants of:

> T=3, d=4, therefore X has shape [3,4].

That is useful once, but repeated generic shape examples create the appearance of explanation without teaching the chapter's mechanism.

Each chapter should have examples specific to the concept.

Examples:

- tokenization: trace an actual Japanese/English string into plausible token pieces and IDs, while making clear that exact tokenization depends on the tokenizer;
- embedding: use a tiny lookup table and show how IDs select rows;
- dot product: compare concrete vectors and show why sign/magnitude matters;
- Q/K/V: compute all three projections from the same `X`;
- Softmax: compare score vectors before and after exponentiation/normalization and show sensitivity to scale;
- causal mask: display the actual mask matrix and show which logits become inaccessible;
- multi-head attention: show how dimensions are partitioned/reshaped and why independent projections permit different learned subspaces without claiming heads have fixed human-readable roles;
- residual connection: numerically add an input vector back to a transformed vector and explain the optimization/information-flow intuition;
- LayerNorm/RMSNorm: use tiny values and explain normalization axes;
- logits: show a small vocabulary, logits, probabilities, and one sampling step.

---

## 6. Q/K/V and Attention require a full worked-through treatment

The Q/K/V and attention chapters are central and should not be considered complete until a reader can answer all of the following from the text itself:

- Why make three projections from the same input?
- What is learned in `W_Q`, `W_K`, and `W_V`?
- Why compare Q with K rather than Q with V?
- What exactly does a dot product represent here?
- Why transpose K?
- What are the rows and columns of `QK^T`?
- Why divide by `sqrt(d_k)`?
- What numerical problem appears without scaling as dimensionality grows?
- Why use Softmax?
- Why multiply the resulting weights by V?
- What changes when attention is self-attention?
- What changes when it is cross-attention?
- What does a causal mask do before Softmax?
- What does an attention output vector contain after mixing?
- Which parts are learned parameters and which are activations computed for the current input?

Include at least one **complete numeric path from X to attention output** with no hidden step.

---

## 7. Mathematics policy

The book must not become either of these extremes:

1. a math textbook that delays Transformer understanding for hundreds of pages;
2. a hand-wavy visual guide that hides every real operation.

Use mathematics **just in time**.

When a mathematical idea first becomes necessary:

1. explain the need;
2. introduce the smallest useful definition;
3. use it immediately inside the Transformer;
4. optionally link to a dedicated math note for deeper treatment.

Explain notation on first use.

Never assume the reader knows what symbols such as `R^d`, transpose, norm, expectation, gradient, Jacobian, or probability distribution mean.

If a derivation is optional, label it optional rather than forcing it into the main path.

Do not require large hand calculations.

---

## 8. Training deserves a mechanism, not a slogan

Do not summarize training as "calculate loss and use backpropagation".

At minimum, explain conceptually and with a tiny example:

- next-token training pairs;
- logits;
- probability of the correct next token;
- cross-entropy / negative log likelihood intuition;
- what a parameter is;
- what it means for loss to change when a parameter changes;
- gradient as local sensitivity/direction, without requiring calculus fluency;
- optimizer update;
- how repeated updates change embeddings and projection matrices;
- distinction between training-time weight updates and inference-time activations;
- why the model does not store a simple database entry for each training sentence.

Use careful language about memorization, generalization, and learned representations.

---

## 9. Modern LLM material must be clearly separated from the 2017 Transformer

Distinguish:

- the original Transformer architecture;
- decoder-only GPT-style language models;
- modern implementation changes and optimizations.

When discussing RoPE, RMSNorm, SwiGLU, grouped-query attention, multi-query attention, FlashAttention, KV cache, MoE, long-context methods, etc., explicitly label them as later developments or common modern variants where appropriate.

Do not accidentally imply that every modern LLM has the same architecture.

Use primary papers / official technical reports where possible.

---

## 10. Interactive components are teaching instruments, not decoration

Keep or improve interactivity only when it lets the reader test a causal relationship.

Good interactive examples:

- edit Q/K vectors and watch dot-product scores change;
- change a Softmax temperature/scale and watch the distribution sharpen or flatten;
- toggle a causal mask and see inaccessible positions disappear;
- change one value in V and see which output positions change;
- compare one head with multiple heads using the same input;
- step token-by-token through generation and inspect the growing KV cache.

Every interaction must have:

- a short instruction: "change X and observe Y";
- an explanation of what the reader should learn from the change;
- a non-interactive textual/table fallback;
- keyboard accessibility where applicable;
- no dependence on animation to communicate essential information.

Do not build interactions merely to satisfy a component count.

---

## 11. Prose quality

Write natural Japanese textbook prose.

Avoid AI-generated template rhythm where every chapter has nearly identical paragraphs with only nouns replaced.

Avoid excessive headings that fragment a simple explanation into tiny pieces.

Prefer cohesive paragraphs that develop an idea over one-sentence paragraph stacks.

Use terminology consistently, but vary explanation style according to the concept.

The prose should feel authored, not filled in from a schema.

Do not overuse phrases equivalent to:

- "まず確認しましょう"
- "大切なのは〜です"
- "つまり〜ということです"
- "声に出して確認"

unless they genuinely help in that specific passage.

---

## 12. Source and accuracy policy

Before rewriting, refresh the technical reference set.

Prefer:

- original research papers;
- official framework/library documentation;
- authoritative technical reports from model developers;
- well-established educational references only as secondary support.

For every chapter, distinguish among:

- established mechanism/fact;
- common implementation choice;
- pedagogical simplification;
- interpretation or intuition.

Do not copy source prose. Write original explanations.

Do not use copyrighted figures unless their license clearly permits reuse. Prefer original SVG/CSS/canvas diagrams generated for this project.

Maintain references in chapter metadata or a consistent citation system.

---

## 13. Continuity across chapters

This should read as one book, not sixteen independent blog posts.

Use a small set of recurring toy examples where continuity genuinely helps, but do not force one example everywhere.

Each chapter should clearly indicate:

- what the reader already knows from earlier chapters;
- what new capability is being added;
- which later chapter will use it.

Avoid re-explaining the same prerequisite from scratch in every chapter. Link backward when appropriate and deepen it only when the new context requires more detail.

---

## 14. Tiny Transformer implementation

The tiny implementation should become the book's executable companion.

Ensure that major pieces in the implementation can be mapped back to the chapter that explains them.

Where useful, add comments or documentation links such as:

- embedding lookup → embedding chapter;
- Q/K/V projections → QKV chapter;
- attention score calculation → attention chapter;
- causal mask → masking chapter;
- residual + norm → block chapter;
- logits → output chapter.

The implementation should favor inspectability over performance.

Do not pretend that a tiny educational model reproduces all behavior of production LLMs.

---

## 15. Tests for content quality

Existing automated checks are not sufficient to declare chapters pedagogically complete.

Add or extend lightweight content-quality checks where they can detect obvious regressions, for example:

- suspiciously short core chapters;
- placeholder code such as `transform(x)` in finished core chapters;
- missing references;
- missing worked examples in designated core chapters;
- duplicate boilerplate paragraphs repeated across many chapters;
- internal links broken under the GitHub Pages base path.

Do not game these tests by inserting meaningless filler.

Human-readable quality remains the primary goal.

---

## 16. Work autonomously in a large batch

This project is being developed during a period where large autonomous Manus runs are preferred.

Do not ask for approval after each chapter.

Routine decisions about prose organization, examples, diagrams, component implementation, test fixes, citations, and documentation should be made independently.

If one chapter reveals that an earlier explanation must change, revise the earlier chapter too.

Continue until the entire book reaches the strongest coherent second-draft state feasible.

Escalate only genuine blockers involving:

- secrets or credentials;
- paid services;
- irreversible external operations;
- unresolved legal/copyright questions;
- a major change to the educational purpose of Transformer Atlas.

---

## 17. Completion criteria

Do **not** report completion merely because all chapter files exist or the Astro build is green.

The second draft is complete only when:

1. every core chapter contains substantive explanatory prose rather than scaffold text;
2. major mechanisms have real worked numeric examples;
3. Q/K/V → attention output is traced end-to-end with actual numbers;
4. formulas are motivated and decoded rather than merely displayed;
5. code examples correspond to the concept being taught;
6. repeated boilerplate has been removed;
7. chapters form a coherent learning progression;
8. original-vs-modern Transformer distinctions are accurate;
9. interactive components teach specific relationships rather than decorate pages;
10. Tiny Transformer maps clearly to the curriculum;
11. citations and provenance are consistent;
12. accessibility and non-interactive fallbacks remain intact;
13. site build/tests are green;
14. GitHub Pages base-path links are valid;
15. the result can reasonably be called a **book**, not a curriculum skeleton.

---

## 18. Final report

After the full pass, provide one consolidated report only.

Include:

- chapters substantially rewritten;
- approximate Japanese character count before/after;
- core worked examples added;
- interactive teaching components added or materially improved;
- math notes added/expanded;
- Tiny Transformer changes;
- source/reference improvements;
- content-quality tests added;
- build/test/deployment status;
- known pedagogical weaknesses that still remain;
- any true owner-only decisions.

Do not present "all files exist" as the primary success metric. The primary success metric is whether a motivated non-specialist can now **follow the mechanism from text to tensors to code**.
