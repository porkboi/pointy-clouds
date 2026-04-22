---
name: math-pdf-summarizer
description: Summarize linked or local math-heavy PDFs into Markdown while preserving the key mathematical content. Use when a prompt includes a PDF URL or PDF file and the task is to extract the main ideas, notation, assumptions, definitions, lemmas, theorems, proofs, derivations, algorithms, or formulas from papers, lecture notes, appendices, or technical reports.
---

# Math PDF Summarizer

## Overview

Turn a math-heavy PDF into a concise Markdown summary that keeps the mathematics intact instead of flattening it into generic prose.

Prefer explicit notation, equation names, assumptions, and result statements. When extraction is noisy, say so and separate confirmed claims from inferred ones.

## Workflow

1. Identify the PDF source.
2. Extract enough text, equations, and structure to understand the paper.
3. Build a Markdown summary around notation, results, and derivations.
4. Flag uncertainty where the PDF rendering or OCR is unreliable.

## Source Handling

### Linked PDF

Use the available web tooling first when the prompt includes an HTTP(S) link. Open the PDF or landing page, confirm that it is the correct document, and inspect the visible text before attempting to summarize.

If the PDF is long, triage it:

- Read title, abstract, introduction, conclusion, and section headings first.
- Jump to theorem, lemma, proposition, algorithm, and experiment sections.
- Inspect appendices only if they carry core derivations or definitions.

### Local PDF

If the user provides a local PDF path, inspect the file directly with local tools available in the environment. Prefer text extraction over OCR when possible.

## Math Extraction Priorities

Capture these in roughly this order:

1. Problem statement and objective.
2. Core notation and variable definitions.
3. Assumptions, constraints, and setup.
4. Main results: theorem statements, propositions, guarantees, convergence claims, bounds.
5. Key equations: objective functions, update rules, likelihoods, losses, recurrences, identities.
6. Derivation flow: what follows from what, and which steps are central.
7. Algorithms: inputs, outputs, major steps, and complexity if stated.
8. Empirical or illustrative conclusions only after the math is covered.

Do not copy large equation blocks verbatim unless required. Re-express them in concise Markdown math and explain what each one does.

## Accuracy Rules

- Preserve symbols and subscripts when they matter.
- Distinguish exact statements from your interpretation.
- State when notation is overloaded or ambiguous in the source.
- Do not invent missing proof steps.
- If extraction corrupts an equation, describe the role of the equation and mark the transcription as uncertain.

For dense derivations, summarize the chain:

- Starting point
- Transformation or approximation used
- Intermediate result
- Final conclusion

## Output Requirements

Write a Markdown file. Use the template in `references/summary-template.md`.

At minimum include:

- Title and source link or file path
- One-paragraph overview
- Glossary of notation
- Main mathematical results
- Key derivations or proof ideas
- Algorithm or method summary if present
- Open uncertainties caused by extraction quality

Prefer short equations inline with explanation. Use display math only for the few equations that are central to the document.

## Working Style

Start by collecting evidence before writing the summary. If the document is long, produce a high-confidence summary of the main text rather than pretending to cover every appendix detail.

When the PDF is especially technical, build the summary section-by-section and keep a running note of notation so symbols remain consistent.

If the prompt asks for "all key math concepts," interpret that as:

- Every central definition
- Every named result or guarantee
- Every objective or update rule needed to understand the method
- Every assumption that changes the meaning of the results

Do not treat implementation trivia or generic background prose as a key math concept unless the paper relies on it.

## Reference Files

- Read `references/summary-template.md` before drafting the final Markdown.
