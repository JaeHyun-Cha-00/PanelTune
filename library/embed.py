import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer

LIBRARY = "library/data"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD = 0.88   # library examples this close in meaning to a test question are removed

def load_test_questions():
    # every benchmark test question, written the same way the library writes them
    questions = []
    for task in ["gsm8k", "mmlu", "humaneval", "ifeval"]:
        for line in open(f"runs/start/samples/{task}.jsonl"):
            q = json.loads(line)["question"]
            if task == "gsm8k":
                questions.append(q["question"])
            elif task == "mmlu":
                choices = "\n".join(f"{letter}. {c}" for letter, c in zip("ABCD", q["choices"]))
                questions.append(q["question"] + "\n" + choices)
            else:
                questions.append(q["prompt"])
    return questions

if __name__ == "__main__":
    # 1. read every library example's question
    ids = []
    questions = []
    for path in sorted(glob.glob(f"{LIBRARY}/*.jsonl")):
        for line in open(path):
            record = json.loads(line)
            question = next(m["content"] for m in record["messages"] if m["role"] == "user")
            ids.append(record["id"])
            questions.append(question)

    print(f"{len(questions)} questions to embed")

    # 2. turn each question into 384 numbers
    model = SentenceTransformer(MODEL, device="mps")
    vectors = model.encode(questions, batch_size=128, normalize_embeddings=True, show_progress_bar=True)

    # 3. meaning-overlap check: find library examples too close to a test question
    test_vectors = model.encode(load_test_questions(), batch_size=128, normalize_embeddings=True)

    closest = np.zeros(len(ids), dtype=np.float32)
    for start in range(0, len(ids), 5000):
        closest[start:start + 5000] = (vectors[start:start + 5000] @ test_vectors.T).max(axis=1)

    too_close = {ids[i] for i in np.where(closest >= THRESHOLD)[0]}
    print(f"{len(too_close)} examples are too close to a test question (>= {THRESHOLD})")

    # 4. remove them from the library files
    for path in sorted(glob.glob(f"{LIBRARY}/*.jsonl")):
        lines = [line for line in open(path) if json.loads(line)["id"] not in too_close]
        with open(path, "w") as f:
            f.writelines(lines)

    with open(f"{LIBRARY}/removed_meaning_overlap.json", "w") as f:
        json.dump(sorted(too_close), f, indent=2)

    # 5. save the numbers, and which example each row belongs to (without the removed ones)
    keep = [i for i, id_ in enumerate(ids) if id_ not in too_close]
    np.save(f"{LIBRARY}/embeddings.npy", vectors[keep].astype(np.float32))
    with open(f"{LIBRARY}/embedding_ids.json", "w") as f:
        json.dump([ids[i] for i in keep], f)

    print(f"saved {len(keep)} vectors of {vectors.shape[1]} numbers each")
