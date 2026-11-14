from paneltune.panel.tools import load_library

TOTAL = 3000   # training examples per turn

def scale(queries, total=TOTAL):
    # the agent's n's may not add up to the total, so stretch or shrink them all by the same amount
    current = sum(item["n"] for item in queries)
    scaled = [{"query": item["query"], "n": round(item["n"] * total / current)} for item in queries]

    # rounding can be off by 1 or 2, so fix the last one to make the total exact
    scaled[-1]["n"] += total - sum(item["n"] for item in scaled)
    return scaled

def select(queries, used_ids):
    # queries: [{"query": "...", "n": 800}, ...] from the recipe
    # used_ids: ids already trained on earlier in this run (never picked again)

    model, vectors, ids, examples = load_library()

    chosen = []
    taken = set(used_ids)

    for item in queries:
        query_vector = model.encode([item["query"]], normalize_embeddings=True)[0]
        order = (vectors @ query_vector).argsort()[::-1]   # closest first

        count = 0
        for i in order:
            if count == item["n"]:
                break
            if ids[i] in taken:
                continue
            chosen.append(examples[ids[i]])
            taken.add(ids[i])
            count += 1

    return chosen

if __name__ == "__main__":
    queries = [
        {"query": "word problems about discounts", "n": 3},
        {"query": "how do plants make food", "n": 2},
    ]

    first = select(queries, used_ids=[])
    print("FIRST TURN:")
    for record in first:
        print("  ", record["id"])

    second = select(queries, used_ids=[record["id"] for record in first])
    print("SECOND TURN (same queries, first turn's examples marked as used):")
    for record in second:
        print("  ", record["id"])
