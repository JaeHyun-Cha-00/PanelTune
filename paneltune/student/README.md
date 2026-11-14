# Student

The small model being improved: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).

## Files

| File | What it does |
|---|---|
| `data.py` | Turns the builder's recipe into 3,000 training examples, by picking the closest unused examples from the library for each search query |
| `train.py` | Full fine-tune with TRL `SFTTrainer` on the Mac (MPS), batch size 2. The builder picks the learning rate and epochs, kept inside the limits from the config |
| `evaluate.py` | Scores a model on the four benchmarks with `lm-evaluation-harness`, and saves the scores and every answer |

## Benchmarks

| Benchmark | Skill | Score |
|---|---|---|
| GSM8K | Math | exact match |
| MMLU | Knowledge | accuracy |
| HumanEval | Coding | pass@1 |
| IFEval | Instruction following | prompt-level strict accuracy |

All four use the model's chat template. `evaluate.py` writes `scores.json` and `samples/<benchmark>.jsonl`, which holds each question, the model's answer, and whether it was correct.

**Start model** (before any training): GSM8K 9.2%, MMLU 25.5%, HumanEval 18.3%, IFEval 32.5%.
