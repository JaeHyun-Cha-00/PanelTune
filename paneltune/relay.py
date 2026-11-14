import json
import os
import re

import paneltune.panel.tools as tools
from paneltune.panel.agents import agent_ask, agent_dict, token_usage
from paneltune.panel.chat import write_opinions, group_chat, summarize
from paneltune.panel.prompts import BUILDER_SYSTEM, BUILDER_TURN, RECIPE_RETRY
from paneltune.student.data import scale, select
from paneltune.student.train import clamp, train
from paneltune.student.evaluate import evaluate

# load model weights one at a time: transformers' threaded loading freezes on the Mac GPU
# when a model is loaded for evaluation after training in the same program
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

START_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
START_DIR = "runs/start"
AGENTS = list(agent_dict)    # [1, 2, 3, 4]
TURNS = len(AGENTS)          # one turn per agent
BENCHMARKS = ["gsm8k", "mmlu", "humaneval", "ifeval"]

# Helpers

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def scores_table(rows):
    # the scores of every version so far, as a text table for the prompts and scores.csv
    lines = ["version," + ",".join(BENCHMARKS)]
    for row in rows:
        lines.append(row["version"] + "," + ",".join(f"{row[b]:.4f}" for b in BENCHMARKS))
    return "\n".join(lines)

def breakdown_text(breakdown):
    # one line per MMLU subject and IFEval rule, e.g. "mmlu anatomy: 0.2963"
    return "\n".join(f"{task} {name}: {score}" for task in breakdown for name, score in breakdown[task].items())

def extract_recipe(reply):
    # find the last {"select" ... in the agent's reply and read the JSON from there
    # (raw_decode reads one JSON object and ignores any text after it)
    starts = [m.start() for m in re.finditer(r'\{\s*"select"', reply)]
    if not starts:
        raise ValueError('no recipe JSON found (it must start with {"select")')

    recipe, _ = json.JSONDecoder().raw_decode(reply[starts[-1]:])
    return recipe

# Step 1: Plan

def plan(builder_id, turn, scores, breakdown, feedback, folder):
    messages = [
        {"role": "system", "content": BUILDER_SYSTEM},
        {"role": "user", "content": BUILDER_TURN.format(turn=turn, scores=scores, breakdown=breakdown, feedback=feedback)},
    ]
    reply = agent_ask(builder_id, messages, tools=tools.BUILDER_TOOLS, rounds=10)
    save_text(f"{folder}/reply.md", reply)

    # if the recipe can't be read, ask once more, without tools
    try:
        recipe = extract_recipe(reply)
    except ValueError as error:
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": RECIPE_RETRY.format(error=error)})
        reply = agent_ask(builder_id, messages)
        save_text(f"{folder}/retry.md", reply)
        recipe = extract_recipe(reply)

    save_json(f"{folder}/recipe.json", recipe)
    return recipe

# The relay

def run(config, solo_agent=None):
    # config: the settings from a configs/*.yaml file
    # mode "panel": agent 1 builds turn 1, agent 2 builds turn 2, ... the others review
    # mode "solo": one agent (solo_agent) builds every turn and reviews its own results

    mode = config["mode"]
    if mode == "panel":
        builders = AGENTS
    else:
        builders = [solo_agent] * TURNS

    run_dir = config["run_dir"].format(agent=solo_agent)
    tools.RUN_DIR = run_dir   # the agents' file and note tools work inside this folder
    save_json(f"{run_dir}/config.json", {**config, "solo_agent": solo_agent})

    # everything starts from the untuned model and its saved scores
    model_path = START_MODEL
    rows = [{"version": "start", **load_json(f"{START_DIR}/scores.json")}]
    breakdown = load_json(f"{START_DIR}/breakdown.json")
    feedback = "None yet. This is the first turn."
    used_ids = []

    for turn in range(1, TURNS + 1):
        builder = builders[turn - 1]
        turn_dir = f"{run_dir}/turn_{turn}"
        tokens_before = {i: dict(usage) for i, usage in token_usage.items()}
        print(f"\n=== turn {turn}: Agent {builder} builds ===", flush=True)

        # 1. plan
        recipe = plan(builder, turn, scores_table(rows), breakdown_text(breakdown), feedback, f"{turn_dir}/1_plan")

        # 2. select (never an example used in an earlier turn)
        examples = select(scale(recipe["select"], config["examples_per_turn"]), used_ids)
        used_ids += [example["id"] for example in examples]
        save_text(f"{turn_dir}/2_data/selected.jsonl", "".join(json.dumps(e) + "\n" for e in examples))

        # 3. train the previous version
        settings = clamp(recipe.get("training", {}), config["learning_rate"], config["epochs"])
        save_json(f"{turn_dir}/3_train/settings.json", settings)
        model_path = train(model_path, examples, f"{turn_dir}/3_train/model", settings["learning_rate"], settings["epochs"])

        # 4. evaluate
        scores = evaluate(model_path, f"{turn_dir}/4_eval", limit=config["eval_limit"])
        breakdown = load_json(f"{turn_dir}/4_eval/breakdown.json")
        rows.append({"version": f"v{turn}", **scores})
        save_text(f"{run_dir}/scores.csv", scores_table(rows) + "\n")

        # 5. discuss (not after the last turn: there is no next builder)
        if turn < TURNS:
            chat_path = f"{turn_dir}/5_discuss/chat.jsonl"
            os.makedirs(os.path.dirname(chat_path), exist_ok=True)

            if mode == "panel":
                reviewers = [agent for agent in AGENTS if agent != builder]
                chat = write_opinions(reviewers, turn, builder, recipe, scores_table(rows), breakdown_text(breakdown), chat_path)
                chat = group_chat(reviewers, chat, chat_path, max_rounds=config["max_chat_rounds"])
                feedback = summarize(builders[turn], chat)
            else:
                chat = write_opinions([builder], turn, builder, recipe, scores_table(rows), breakdown_text(breakdown), chat_path)
                feedback = chat[0]["text"]

            save_text(f"{turn_dir}/5_discuss/feedback.md", feedback)

        # tokens used by each agent this turn
        tokens = {i: {k: token_usage[i][k] - tokens_before[i][k] for k in usage} for i, usage in token_usage.items()}
        save_json(f"{turn_dir}/tokens.json", tokens)

    print(f"\n=== done ===\n{scores_table(rows)}", flush=True)
    return rows
