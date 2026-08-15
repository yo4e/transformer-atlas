# Transformer Atlas — Manus Build Brief

## Mission

Build `yo4e/transformer-atlas` into a complete, technically accurate, interactive Japanese learning site about Transformer architecture.

Read first:

- `README.md`
- `docs/DESIGN.md`

Use `docs/DESIGN.md` as the main curriculum and product specification. Improve it when primary-source research shows a clear reason.

This is a large end-to-end assignment. Continue through research, curriculum validation, Astro implementation, full-course writing, interactive diagrams, code examples, testing, accessibility, documentation, and deployment readiness without stopping after each small step for approval.

## Educational goal

The site is for an adult non-specialist who can follow middle-school mathematics but should not be assumed fluent in high-school or university mathematics.

The learning path should repeatedly bridge:

**intuition -> actual mechanism -> minimal mathematics -> tiny example -> code -> complete architecture**

Do not make the prose childish. Do not hide mechanisms behind metaphors. Explain new mathematical objects at first use and show why they are needed in the Transformer.

## Research and source policy

Before and during writing, verify technical claims using authoritative sources.

Prefer:

- original Transformer and architecture papers;
- primary papers for RoPE, RMSNorm, attention variants, GQA/MQA, MoE, and other modern variations when discussed;
- official framework documentation for implementation details.

Create a durable research area such as:

```text
docs/research/
  SOURCES.md
  CURRICULUM_REVIEW.md
  MISCONCEPTIONS.md
```

Record what each source supports and where it is used. Distinguish original-Transformer facts, common modern practices, implementation-specific choices, and pedagogical simplifications.

Do not copy textbook/course prose or diagrams. Write original explanations and diagrams based on cited sources.

## Build the real book

Do not stop with an outline or sample chapters. Produce the complete first-edition chapter set described in `docs/DESIGN.md` as far as feasible.

Each substantive chapter should contain, where appropriate:

- the problem being solved;
- intuitive explanation;
- actual computational mechanism;
- minimal necessary mathematics;
- every symbol defined;
- tiny worked example;
- code or pseudocode;
- explanation of where a metaphor stops being literal;
- connection to real LLMs;
- 2–4 conceptual comprehension questions;
- concise takeaway;
- references.

Avoid padding. Depth should come from accurate explanation, examples, diagrams, code, and connections between concepts.

## Mathematics policy

When vectors, matrices, transpose, dot product, softmax, exponentials, probability, derivatives, gradients, or the chain rule first appear:

1. explain the concept in ordinary language;
2. explain why it is needed at that point;
3. show a very small numeric example;
4. introduce the notation;
5. connect it to code.

Create optional math reference pages so readers do not have to finish a long prerequisite course before seeing Transformer architecture.

Show tensor/matrix shapes whenever they materially aid understanding.

## Accuracy guardrails

Avoid common misleading simplifications. In particular:

- tokens are not simply words;
- embedding dimensions do not normally have simple human-readable labels;
- attention heads should not be casually described as fixed human concepts such as “grammar heads”;
- attention weights are not a universal explanation of model reasoning;
- softmax outputs are not automatically epistemic confidence;
- architecture, training data, post-training, RAG, tools, and agent systems are different layers;
- synthetic or illustrative attention visualizations must be labeled as such;
- original sinusoidal positional encoding should be distinguished from modern RoPE-based systems;
- do not imply every modern model uses every modern architectural variation.

When simplifying, state what was simplified.

## Site implementation

Use Astro unless a concrete technical blocker appears.

Preferred architecture:

- static-first Astro site;
- Markdown/MDX chapters;
- TypeScript;
- client-side islands only for useful interactions;
- static math rendering;
- syntax-highlighted code;
- structured content collections/frontmatter;
- previous/next navigation;
- part overview pages;
- concept index;
- glossary;
- math reference index;
- search if practical without heavy infrastructure;
- no database;
- no account requirement;
- no mandatory hosted model/API;
- deployable as a static site.

Keep content separate enough from presentation that the book could later be reused for PDF/EPUB/print workflows.

## Interactive atlas components

Implement interactions only when they improve understanding. Prioritize:

- tokenizer explorer;
- vector/embedding explorer;
- dot-product lab;
- attention sentence visualization;
- step-by-step Q/K/V pipeline;
- softmax sandbox;
- causal-mask matrix;
- Transformer block flow diagram;
- temperature/top-k/top-p sampling explorer;
- KV-cache comparison;
- tiny-model forward-pass trace.

Not every chapter needs a custom component. Every interactive view should have accompanying prose and an accessible fallback explanation.

## Visual and accessibility requirements

Aim for a clean technical-atlas aesthetic. Prioritize Japanese reading typography, diagrams, formulas, matrices, code, and annotations. Avoid decorative “AI neon” visual language that does not aid learning.

Accessibility requirements include:

- semantic HTML;
- keyboard-operable controls;
- visible focus states;
- reduced-motion support;
- textual alternatives for diagrams;
- adequate contrast;
- no color-only meaning;
- responsive tables/matrices;
- reasonable screen-reader support for math and interactive content.

## Tiny Transformer implementation

Create a small educational decoder-only Transformer near the end of the curriculum.

Requirements:

- CPU-friendly;
- deterministic where practical;
- small enough to inspect;
- tensor shapes documented;
- attention mechanics not hidden behind one opaque high-level call;
- major code sections cross-linked to the chapters that explain them;
- teaching implementation clearly separated from production-performance claims.

Python + PyTorch is a reasonable default if it provides the clearest result, but choose the final implementation pragmatically.

## Testing and editorial QA

Set up appropriate automated checks for:

- Astro build;
- TypeScript;
- content/frontmatter validity;
- internal links;
- code examples where practical;
- interactive components;
- CI.

Also create and apply an editorial/technical checklist covering:

- undefined mathematical symbols;
- concepts used before introduction;
- inconsistent terminology;
- code/prose mismatches;
- tensor-shape mistakes;
- misleading metaphors;
- unsupported technical claims;
- broken cross-links;
- unlabeled simplifications.

## Repository hygiene

Maintain or add as appropriate:

- README;
- explicit license;
- CONTRIBUTING.md;
- content/style documentation;
- research/source provenance;
- CI;
- deployment instructions;
- glossary conventions;
- architecture notes.

Keep the repository understandable to another maintainer.

## Deployment readiness

Prepare a static deployment configuration suitable for a straightforward host such as GitHub Pages or Cloudflare Pages. Do not require a paid service.

If actual deployment requires account-specific authorization, leave the project deployment-ready and document the final owner step.

## Work in large chunks

Do not stop after routine milestones with messages such as “Chapter 1 is finished, should I continue?” or “The scaffold is ready, should I write the content?”

Continue autonomously through all independent work. Make routine decisions about libraries, components, filenames, CSS, tests, diagrams, wording, and minor curriculum adjustments yourself.

If something fails, investigate, repair, and try another reasonable approach before escalating.

Owner input is only needed for genuine blockers such as credentials, irreversible external operations, unresolved licensing/copyright questions, or a major change to the educational purpose.

If a blocker occurs, report what happened, what was tried, available options, the recommended option, and what work can continue meanwhile.

## Scope discipline

Do not expand this into a general machine-learning curriculum. Brief historical contrast with RNNs/CNNs is fine, but full courses on linear algebra, calculus, probability, distributed training, GPU kernels, RAG, agents, or commercial LLM APIs are outside the first release unless directly necessary to understand the Transformer.

## Completion criteria

Push toward a state where a third party can clone/deploy the repository and use it as a real learning resource.

A strong completion state includes:

1. curriculum validated against authoritative sources;
2. coherent first-edition chapter set;
3. math side-pages for required concepts;
4. functional Astro site;
5. meaningful interactive diagrams for difficult concepts;
6. tiny educational Transformer implementation;
7. cross-links between prose, equations, diagrams, and code;
8. technical/editorial fact-check pass;
9. responsive and accessible layout;
10. green build/CI;
11. repository documentation and source provenance;
12. deployment-ready configuration;
13. known limitations and future work documented.

Do not stop at “research complete”, “scaffold complete”, or “sample chapters complete” if further progress is feasible.

## Final report

When the strongest coherent state is reached, provide one consolidated report covering:

- research completed;
- curriculum changes made after research;
- chapters/content completed;
- interactive components completed;
- tiny-model status;
- source/provenance approach;
- site architecture;
- accessibility work;
- tests/build/CI status;
- deployment status;
- major commits/PRs;
- known limitations;
- owner-only decisions that remain;
- the recommended learner starting path.

The result should be a technically trustworthy, readable, interactive book that genuinely helps someone understand Transformer architecture.
