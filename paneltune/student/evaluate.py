import os
import json
from lm_eval import simple_evaluate

SCORE_FIELD = {
    "gsm8k": "exact_match",
    "mmlu": "acc",
    "humaneval": "pass@1",
    "ifeval": "prompt_level_strict_acc",
}

def evaluate(model_path, output_dir, limit=None):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=["gsm8k", "mmlu", "humaneval", "ifeval"],
        device="mps",
        batch_size=8,
        limit=limit,
        apply_chat_template=True,
        log_samples=True,
        confirm_run_unsafe_code=True,
    )

    scores = {
        "gsm8k": results["results"]["gsm8k"]["exact_match,flexible-extract"],
        "mmlu": results["results"]["mmlu"]["acc,none"],
        "humaneval": results["results"]["humaneval"]["pass@1,create_test"],
        "ifeval": results["results"]["ifeval"]["prompt_level_strict_acc,none"],
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "scores.json")
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)

    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    mmlu_subjects = [name for name in results["samples"] if name.startswith("mmlu_")]
    sample_groups = {
        "gsm8k": ["gsm8k"],
        "mmlu": mmlu_subjects,
        "humaneval": ["humaneval"],
        "ifeval": ["ifeval"],
    }

    for task, names in sample_groups.items():
        path = os.path.join(samples_dir, f"{task}.jsonl")
        with open(path, "w") as f:
            for name in names:
                for sample in results["samples"][name]:
                    if task == "gsm8k" and sample["filter"] != "flexible-extract":
                        continue

                    if task == "mmlu":
                        option_scores = [score for score, _ in sample["filtered_resps"]]
                        response = option_scores.index(max(option_scores))
                    else:
                        response = sample["resps"][0][0]

                    record = {
                        "question": sample["doc"],
                        "target": sample["target"],
                        "response": response,
                        "correct": sample[SCORE_FIELD[task]],
                    }
                    f.write(json.dumps(record) + "\n")

    return scores

if __name__ == "__main__":
    scores = evaluate("HuggingFaceTB/SmolLM2-360M-Instruct", "runs/start", limit=None)
