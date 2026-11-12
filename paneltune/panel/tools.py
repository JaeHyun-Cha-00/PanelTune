import glob
import json
import os
from functools import cache

RUN_DIR = "runs/test_run"
LIBRARY_DIR = "library/data"

# File Tools

def list_files():

    names = []
    for folder, subfolders, files in os.walk(RUN_DIR):
        subfolders[:] = [s for s in subfolders if s not in ("model", "samples")]
        for file in files:
            path = os.path.join(folder, file)
            names.append(os.path.relpath(path, RUN_DIR))

    if not names:
        return "No files yet."

    return "\n".join(sorted(names))

def read_file(name):

    run_dir = os.path.abspath(RUN_DIR)
    path = os.path.abspath(os.path.join(RUN_DIR, name))

    if os.path.commonpath([run_dir, path]) != run_dir:
        return "Error: you can only read files inside the run folder."

    # samples/ holds benchmark test questions, model/ holds model weights: both are hidden
    folders = os.path.relpath(path, run_dir).split(os.sep)
    if "samples" in folders or "model" in folders:
        return "Error: this folder is hidden."

    with open(path, "r") as f:
        return f.read()

# Note Tools

def write_note(agent_id, text):

    notes_dir = os.path.join(RUN_DIR, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    path = os.path.join(notes_dir, f"agent_{agent_id}.txt")

    with open(path, "a") as f:
        f.write(text + "\n")

    return "Noted Saved"

def read_note(agent_id):
    
    notes_dir = os.path.join(RUN_DIR, "notes")
    path = os.path.join(notes_dir, f"agent_{agent_id}.txt")

    if not os.path.exists(path):
        return "No Notes"

    with open(path, "r") as f:
        return f.read()

# Library Tools

@cache
def load_library():
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = np.load(f"{LIBRARY_DIR}/embeddings.npy")
    ids = json.load(open(f"{LIBRARY_DIR}/embedding_ids.json"))

    examples = {}
    for path in glob.glob(f"{LIBRARY_DIR}/*.jsonl"):
        for line in open(path):
            record = json.loads(line)
            examples[record["id"]] = record

    return model, vectors, ids, examples

def search_library(query, n=5):

    model, vectors, ids, examples = load_library()

    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = vectors @ query_vector
    top = scores.argsort()[::-1][:min(n, 10)]

    results = []
    for i in top:
        record = examples[ids[i]]
        question = record["messages"][-2]["content"][:300]
        answer = record["messages"][-1]["content"][:300]
        results.append(f"[{record['id']}] ({record['skill']})\nQ: {question}\nA: {answer}")

    return "\n\n".join(results)

# Tool Definitions (what the model sees)

LIST_FILES = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List the files saved so far in this run, such as past scores, recipes and feedback.",
        "parameters": {"type": "object", "properties": {}},
    },
}

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read one file from this run. Use a name exactly as list_files shows it, e.g. turn_1/4_eval/scores.json.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The file name from list_files."}},
            "required": ["name"],
        },
    },
}

WRITE_NOTE = {
    "type": "function",
    "function": {
        "name": "write_note",
        "description": "Save a note to your personal notebook. Your notes stay available in later turns.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The note to save."}},
            "required": ["text"],
        },
    },
}

READ_NOTE = {
    "type": "function",
    "function": {
        "name": "read_note",
        "description": "Read all the notes in your personal notebook from earlier turns.",
        "parameters": {"type": "object", "properties": {}},
    },
}

SEARCH_LIBRARY = {
    "type": "function",
    "function": {
        "name": "search_library",
        "description": (
            "Search the training-data library by meaning. Describe the kind of examples you want in plain words, "
            "e.g. 'multi-step word problems about discounts' or 'answer in exactly 3 bullet points'. "
            "Returns the closest examples with their id, skill, question and answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What kind of examples you are looking for."},
                "n": {"type": "integer", "description": "How many examples to return (1 to 10). Default 5."},
            },
            "required": ["query"],
        },
    },
}

BUILDER_TOOLS = [LIST_FILES, READ_FILE, READ_NOTE, SEARCH_LIBRARY]
REVIEWER_TOOLS = [LIST_FILES, READ_FILE, WRITE_NOTE, READ_NOTE]

TOOL_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_note": write_note,
    "read_note": read_note,
    "search_library": search_library,
}

NEEDS_AGENT_ID = {"write_note", "read_note"}

if __name__ == "__main__":
    print(search_library("word problems about discounts and percentages", 3))
    print("\n==========\n")
    print(search_library("write a function that checks if a number is prime", 3))
