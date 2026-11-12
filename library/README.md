# Library

The shared training-data library. Builders pick their training examples from here.

**104,368 examples** from 23 public datasets (train splits only), at most 5,000 per dataset.

## Datasets

| Skill | Datasets |
|---|---|
| Math | gsm8k, orca_math, svamp, aqua_rat, hendrycks_math |
| Knowledge | arc, openbookqa, sciq, commonsense_qa, qasc |
| Coding | self_oss_instruct, mbpp, magicoder, code_alpaca, python_code_18k |
| Instructions | tulu3_if, ifeval_like, no_robots, alpaca, dolly |
| General | ultrachat, slimorca, wizardlm |

## Format

Every dataset is converted to the same chat format, one example per line:

```json
{"id": "gsm8k_00042", "source": "openai/gsm8k", "skill": "math", "messages": [...]}
```

Multiple-choice questions are written like MMLU: the question, options A to D, and the answer letter.

## Cleaning

- Drop empty examples, examples over 1,024 tokens, and duplicate questions.
- **Word overlap:** drop any example that shares a 13-word sequence with a benchmark test question.
- **Meaning overlap:** drop any example whose embedding is very close to a test question (cosine 0.88 or more), to catch reworded copies.

## Files

| File | What it does |
|---|---|
| `build.py` | Downloads, converts and cleans the datasets, and saves `data/<dataset>.jsonl` |
| `embed.py` | Embeds every question with `all-MiniLM-L6-v2`, removes meaning overlaps, and saves the vectors for search |
| `data/` | The built library (git-ignored) |