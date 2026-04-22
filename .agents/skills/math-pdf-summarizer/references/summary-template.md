# Markdown Summary Template

## Use

Fill this template after extracting the paper structure and notation. Keep the finished summary concise, but do not omit core mathematical objects or claims.

## Template

```md
# {Paper title}

Source: {URL or local file path}

## Overview

{2-5 sentence summary of the paper's objective, method, and main mathematical contribution.}

## Problem Setup

- Goal: {What problem is being solved?}
- Inputs: {Data, variables, assumptions, domains}
- Output: {Estimator, policy, proof, embedding, optimizer, bound, etc.}

## Notation

| Symbol | Meaning |
| --- | --- |
| `{x}` | {Meaning} |
| `{f(x)}` | {Meaning} |

## Core Definitions

- **{Definition name}**: {Concise explanation}

## Main Results

- **{Theorem / proposition / claim}**: {Statement in plain language plus the key bound, guarantee, or condition}

## Key Equations

1. `{equation or compact math form}`: {Role of the equation}
2. `{equation or compact math form}`: {Role of the equation}

## Derivation Flow

1. {Starting point}
2. {Main transformation, inequality, approximation, or decomposition}
3. {Intermediate result}
4. {Final implication}

## Method or Algorithm

1. {Main step}
2. {Main step}
3. {Main step}

## Assumptions and Limits

- {Assumption}
- {Limitation or condition for validity}

## Uncertainties

- {Equation, symbol, or section that could not be recovered cleanly}
```

## Notes

- Keep notation consistent with the paper unless the original symbols are unreadable.
- Prefer concise mathematical paraphrase over long verbatim extraction.
- If the paper has no formal theorem statements, replace `Main Results` with the strongest mathematical claims the paper actually makes.
