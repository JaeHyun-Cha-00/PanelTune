# PanelTune

A panel of four small AI agents takes turns fine-tuning an even smaller language model.

## 1. The Question

**Can a panel of agents improve a small model better than a single teacher?**

Small language models are cheap and run anywhere, but they fall far behind frontier models. One common way to improve them is to have a stronger model act as a teacher: it decides what training data to use and how to train.

PanelTune tests a different idea: instead of one teacher, four small agents take turns improving the student, and the result is compared against each agent working alone.

## 2. How It Works

PanelTune runs as a **relay** of four turns. In each turn, one agent builds and the other three review:

1. **Build.** The builder reads the scores and feedback so far, picks training data from a shared library, then fine-tunes the student on 3,000 examples.
2. **Benchmark.** The new version is scored on four benchmarks.
3. **Discuss.** The other three agents review the scores and failures in a group chat, and pass feedback to the next builder.

**Baseline:** each agent also runs all four turns **alone**, with the same tools and the same training budget.

## 3. Setup

**Student** ([`paneltune/student/`](paneltune/student/)): [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), fully fine-tuned on a Mac.

**Agents** ([`paneltune/panel/`](paneltune/panel/)): four small models: Qwen3.5 4B, Gemma 4 E4B, Llama 3.2 3B, Ministral 3B.

**Benchmarks** ([`paneltune/student/`](paneltune/student/)): GSM8K (math), MMLU (knowledge), HumanEval (coding), IFEval (instruction following).

**Library** ([`library/`](library/)): 23 public datasets (train splits only), cleaned and checked against the benchmark test questions, about 104k examples.

## 4. Results

TBD

## 5. Getting Started

**1. Install:**

```bash
uv sync
```

**2. Add API keys.** Create a `.env` file in the project folder:

```
DEEPINFRA_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

**3. Evaluate the base model** (the contamination check needs these test questions):

```bash
uv run python -m paneltune.student.evaluate
```

**4. Build the library:**

```bash
uv run python library/build.py
uv run python library/embed.py
```

**5. Run the relay:** TBD

## Notes

Planned for later, not in the first version:
- **Generate + Verify:** let the builder write new training examples, then check them (empty, too long, duplicate, contamination) before training.
- **`run_python` tool:** for checking generated math answers and code.
