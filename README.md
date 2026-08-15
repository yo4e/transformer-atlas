# Transformer Atlas

**Transformerを、式の意味から理解する。**

Transformer Atlas is a Japanese, interactive learning site for understanding how Transformer-based language models work without assuming advanced mathematics.

The goal is not to train readers to solve long algebra exercises. It is to make the machinery legible: why vectors are useful, why dot products appear in attention, what Query / Key / Value are doing, why softmax is needed, how a Transformer block fits together, and how that eventually becomes GPT-style next-token prediction.

## Product idea

The site should combine:

- a coherent book-length curriculum;
- one Markdown / MDX chapter per topic;
- interactive diagrams and small experiments;
- minimal but honest mathematics;
- executable or inspectable code examples;
- short comprehension checks;
- a final tiny Transformer implementation that links each code section back to the relevant chapter.

The intended experience is closer to an **interactive atlas** than a static textbook: readers should be able to move between concepts, formulas, diagrams, and code.

## Target reader

The initial curriculum is designed for an adult learner who:

- can follow middle-school mathematics;
- does not want advanced mathematics treated as prerequisite knowledge;
- values conceptual and architectural understanding over hand-calculation speed;
- wants every new mathematical object explained when it first appears;
- wants to understand what a formula is *for*, not merely memorize it;
- is comfortable reading ordinary technical prose and simple code when carefully explained.

## Planned stack

The initial implementation direction is:

- Astro;
- Markdown / MDX content;
- lightweight browser-side interactive visualizations;
- static-first deployment;
- no mandatory account, API key, hosted model, or paid service.

Exact implementation details may change after technical validation.

## Project documents

- [`docs/DESIGN.md`](./docs/DESIGN.md) — curriculum, UX, content, and technical design
- [`docs/MANUS_BRIEF.md`](./docs/MANUS_BRIEF.md) — autonomous build brief for Manus

## Status

Early design / build phase.
