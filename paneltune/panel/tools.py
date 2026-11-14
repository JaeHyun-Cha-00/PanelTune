import glob
import json
import os
from functools import cache
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

RUN_DIR = None
LIBRARY_DIR = "library/data"

# File Tools

@tool
def list_files() -> str:
    """List the files saved so far in this run, such as past scores, recipes, validation results and feedback."""

    names = []
    for folder, subfolders, files in os.walk(RUN_DIR):
        subfolders[:] = [s for s in subfolders if s not in ("model", "samples")]
        for file in files:
            path = os.path.join(folder, file)
            names.append(os.path.relpath(path, RUN_DIR))

    if not names:
        return "No files yet."

    return "\n".join(sorted(names))

@tool(parse_docstring=True)
def read_file(name: str) -> str:
    """Read one file from this run.

    Args:
        name: The file name exactly as list_files shows it, e.g. turn_1/4_eval/scores.json.
    """

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

@dataclass
class Context:
    agent_id: int

@tool(parse_docstring=True)
def write_note(text: str, runtime: ToolRuntime[Context]) -> str:
    """Save a note to your personal notebook. Your notes stay available in later turns.

    Args:
        text: The note to save.
    """

    notes_dir = os.path.join(RUN_DIR, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    path = os.path.join(notes_dir, f"agent_{runtime.context.agent_id}.txt")

    with open(path, "a") as f:
        f.write(text + "\n")

    return "Noted Saved"

@tool
def read_note(runtime: ToolRuntime[Context]) -> str:
    """Read all the notes in your personal notebook from earlier turns."""

    notes_dir = os.path.join(RUN_DIR, "notes")
    path = os.path.join(notes_dir, f"agent_{runtime.context.agent_id}.txt")

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

@tool(parse_docstring=True)
def search_library(query: str, n: int=5) -> str:
    """Search the training-data library by meaning.

    Describe the kind of examples you want in plain words, e.g. 'multi-step word problems
    about discounts' or 'answer in exactly 3 bullet points'. Returns the closest examples
    with their id, skill, question and answer.

    Args:
        query: What kind of examples you are looking for.
        n: How many examples to return (1 to 10).
    """

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

BUILDER_TOOLS = [list_files, read_file, read_note, search_library]
REVIEWER_TOOLS = [list_files, read_file, write_note, read_note]

if __name__ == "__main__":
    pass
