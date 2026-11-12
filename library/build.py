import json
import os
import random
import re
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from transformers import AutoTokenizer

POOL_SIZE = 20000
MAX_TOKENS = 1024
NGRAM = 13
MAX_PER_DATASET = 5000

# Helpers

def chat(user, assistant):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]

def multiple_choice(question, answer, wrong):
    rng = random.Random(question)
    choices = [answer] + rng.sample(wrong, min(3, len(wrong)))
    rng.shuffle(choices)

    letters = "ABCD"
    lines = [question] + [f"{letters[k]}. {choice}" for k, choice in enumerate(choices)]
    k = choices.index(answer)
    return chat("\n".join(lines), f"{letters[k]}. {answer}")

def labeled_choice(question, choices, answer_key):
    answer = choices["text"][choices["label"].index(answer_key)]
    wrong = [text for text in choices["text"] if text != answer]
    return multiple_choice(question, answer, wrong)

def instruction(instr, extra, output):
    user = instr if not extra else f"{instr}\n\n{extra}"
    return chat(user, output)

# Math

def convert_gsm8k(row):
    return chat(row["question"], re.sub(r"<<.*?>>", "", row["answer"]))

def convert_orca_math(row):
    return chat(row["question"], row["answer"])

def convert_svamp(row):
    answer = str(row["Answer"]).removesuffix(".0")
    return chat(row["question_concat"], f"{row['Equation']} = {answer}\n#### {answer}")

def convert_aqua(row):
    user = row["question"] + "\n" + "\n".join(row["options"])

    # AQuA explanations often end with their own answer line ("ANSWER:B", "Option B", ...)
    # remove it, so every example ends with the same "Answer: B"
    lines = row["rationale"].strip().split("\n")
    if len(lines[-1].strip()) <= 30 and row["correct"] in lines[-1]:
        lines = lines[:-1]

    return chat(user, "\n".join(lines).strip() + f"\nAnswer: {row['correct']}")

def convert_math(row):
    return chat(row["problem"], row["solution"])

# Knowledge

def convert_sciq(row):
    wrong = [row["distractor1"], row["distractor2"], row["distractor3"]]
    return multiple_choice(row["question"], row["correct_answer"], wrong)

def convert_arc(row):
    return labeled_choice(row["question"], row["choices"], row["answerKey"])

def convert_openbookqa(row):
    return labeled_choice(row["question_stem"], row["choices"], row["answerKey"])

def convert_commonsense_qa(row):
    return labeled_choice(row["question"], row["choices"], row["answerKey"])

def convert_qasc(row):
    return labeled_choice(row["question"], row["choices"], row["answerKey"])

# Coding

def convert_self_oss(row):
    return chat(row["instruction"], row["response"])

def convert_mbpp(row):
    tests = "\n".join(row["test_list"])
    user = f"{row['text']}\nYour code should pass these tests:\n{tests}"
    return chat(user, row["code"].replace("\r\n", "\n"))

def convert_magicoder(row):
    if row["lang"] != "python":
        return None
    return chat(row["problem"], row["solution"])

def convert_alpaca_style(row):
    return instruction(row["instruction"], row["input"], row["output"])

# Instructions

def convert_messages(row):
    return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]

def convert_ifeval_like(row):
    return chat(row["prompt"], row["response"])

def convert_dolly(row):
    return instruction(row["instruction"], row["context"], row["response"])

# General / chat

def convert_slimorca(row):
    roles = {"system": "system", "human": "user", "gpt": "assistant"}
    return [{"role": roles[m["from"]], "content": m["value"]} for m in row["conversations"]]

def convert_wizardlm(row):
    return chat(row["instruction"], row["output"])

CATALOG = [
    # (short name, Hugging Face name, config, split, skill, converter)
    ("gsm8k", "openai/gsm8k", "main", "train", "math", convert_gsm8k),
    ("orca_math", "microsoft/orca-math-word-problems-200k", None, "train", "math", convert_orca_math),
    ("svamp", "ChilleD/SVAMP", None, "train", "math", convert_svamp),
    ("aqua_rat", "deepmind/aqua_rat", "raw", "train", "math", convert_aqua),
    ("hendrycks_math", "EleutherAI/hendrycks_math", "ALL", "train", "math", convert_math),

    ("arc", "allenai/ai2_arc", ["ARC-Easy", "ARC-Challenge"], "train", "knowledge", convert_arc),
    ("openbookqa", "allenai/openbookqa", "main", "train", "knowledge", convert_openbookqa),
    ("sciq", "allenai/sciq", None, "train", "knowledge", convert_sciq),
    ("commonsense_qa", "tau/commonsense_qa", None, "train", "knowledge", convert_commonsense_qa),
    ("qasc", "allenai/qasc", None, "train", "knowledge", convert_qasc),

    ("self_oss_instruct", "bigcode/self-oss-instruct-sc2-exec-filter-50k", None, "train", "coding", convert_self_oss),
    ("mbpp", "google-research-datasets/mbpp", "full", "train", "coding", convert_mbpp),
    ("magicoder", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", "coding", convert_magicoder),
    ("code_alpaca", "sahil2801/CodeAlpaca-20k", None, "train", "coding", convert_alpaca_style),
    ("python_code_18k", "iamtarun/python_code_instructions_18k_alpaca", None, "train", "coding", convert_alpaca_style),

    ("tulu3_if", "allenai/tulu-3-sft-personas-instruction-following", None, "train", "instructions", convert_messages),
    ("ifeval_like", "argilla/ifeval-like-data", "filtered", "train", "instructions", convert_ifeval_like),
    ("no_robots", "HuggingFaceH4/no_robots", None, "train", "instructions", convert_messages),
    ("alpaca", "tatsu-lab/alpaca", None, "train", "instructions", convert_alpaca_style),
    ("dolly", "databricks/databricks-dolly-15k", None, "train", "instructions", convert_dolly),

    ("ultrachat", "HuggingFaceH4/ultrachat_200k", None, "train_sft", "general", convert_messages),
    ("slimorca", "Open-Orca/SlimOrca", None, "train", "general", convert_slimorca),
    ("wizardlm", "WizardLMTeam/WizardLM_evol_instruct_70k", None, "train", "general", convert_wizardlm),
]

# Cleaning

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

def is_empty(messages):
    return any(not message["content"].strip() for message in messages)

def is_too_long(messages):
    tokens = tokenizer.apply_chat_template(messages, tokenize=True)["input_ids"]
    return len(tokens) > MAX_TOKENS

seen_questions = set()

def is_duplicate(messages):
    question = next(m["content"] for m in messages if m["role"] == "user")
    question = " ".join(question.lower().split())
    if question in seen_questions:
        return True
    seen_questions.add(question)
    return False

def words(text):
    return re.findall(r"\w+", text.lower())

def ngrams(tokens):
    return {" ".join(tokens[i:i + NGRAM]) for i in range(len(tokens) - NGRAM + 1)}

def load_test_ngrams():
    # every benchmark test question, from the saved Start evaluation
    questions = []
    for task in ["gsm8k", "mmlu", "humaneval", "ifeval"]:
        for line in open(f"runs/start/samples/{task}.jsonl"):
            q = json.loads(line)["question"]
            if task == "gsm8k":
                questions.append(q["question"])
            elif task == "mmlu":
                questions.append(q["question"] + " " + " ".join(q["choices"]))
            else:
                questions.append(q["prompt"])

    # keep only 13-word sequences that belong to exactly one test question
    # (sequences shared by several questions are template text, not a leak)
    count = {}
    for question in questions:
        for gram in ngrams(words(question)):
            count[gram] = count.get(gram, 0) + 1
    return {gram for gram, n in count.items() if n == 1}

test_ngrams = load_test_ngrams()

def is_contaminated(messages):
    text = " ".join(m["content"] for m in messages)
    return not ngrams(words(text)).isdisjoint(test_ngrams)

def load_train(source, config, split):
    if config == "ALL":
        config = get_dataset_config_names(source)
    if isinstance(config, list):
        return concatenate_datasets([load_dataset(source, c, split=split) for c in config])
    return load_dataset(source, config, split=split)

if __name__ == "__main__":
    os.makedirs("library/data", exist_ok=True)

    for short, source, config, split, skill, convert in CATALOG:
        ds = load_train(source, config, split)
        ds = ds.shuffle(seed=0).select(range(min(len(ds), POOL_SIZE)))

        # 1. clean
        clean = []
        dropped = {"empty": 0, "long": 0, "duplicate": 0, "contaminated": 0}
        for row in ds:
            messages = convert(row)
            if messages is None:
                continue

            if is_empty(messages):
                dropped["empty"] += 1
                continue

            if is_too_long(messages):
                dropped["long"] += 1
                continue

            if is_duplicate(messages):
                dropped["duplicate"] += 1
                continue

            if is_contaminated(messages):
                dropped["contaminated"] += 1
                continue

            clean.append(messages)

        # 2. sample down to MAX_PER_DATASET
        if len(clean) > MAX_PER_DATASET:
            clean = random.Random(0).sample(clean, MAX_PER_DATASET)

        # 3. save
        with open(f"library/data/{short}.jsonl", "w") as f:
            for i, messages in enumerate(clean):
                record = {
                    "id": f"{short}_{i:05d}",
                    "source": source,
                    "skill": skill,
                    "messages": messages,
                }
                f.write(json.dumps(record) + "\n")

        print(f"{short}: {len(clean)} saved | dropped {dropped}", flush=True)
