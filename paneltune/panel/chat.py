import json
from paneltune.panel.agents import agent_ask, agent_dict
from paneltune.panel.prompts import REVIEWER_SYSTEM, OPINION, CHAT_ROUND, SUMMARIZE
from paneltune.panel.tools import REVIEWER_TOOLS

def speaker(agent_id):

    return f"Agent {agent_id} ({agent_dict[agent_id]['name']})"

def format_chat(chat):

    return "\n\n".join(f"{speaker(message['agent'])}: {message['text']}" for message in chat)

def save_message(path, message):

    with open(path, "a") as f:
        f.write(json.dumps(message) + "\n")

def write_opinions(reviewer_ids, turn, builder_id, recipe, scores, breakdown, path):

    prompt = OPINION.format(
        turn=turn,
        builder=speaker(builder_id),
        recipe=json.dumps(recipe, indent=2),
        scores=scores,
        breakdown=breakdown,
    )

    chat = []
    for agent_id in reviewer_ids:
        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = agent_ask(agent_id, messages, tools=REVIEWER_TOOLS)
        message = {"round": 0, "agent": agent_id, "text": text}
        chat.append(message)
        save_message(path, message)

    return chat

def is_done(text):
    
    return text.strip(" .!*\n").upper() in ("DONE", "")

def group_chat(reviewer_ids, chat, path, max_rounds=20):

    for round_number in range(1, max_rounds + 1):
        done = 0

        for agent_id in reviewer_ids:
            messages = [
                {"role": "system", "content": REVIEWER_SYSTEM},
                {"role": "user", "content": CHAT_ROUND.format(name=speaker(agent_id), chat=format_chat(chat))},
            ]
            text = agent_ask(agent_id, messages, tools=REVIEWER_TOOLS)

            if is_done(text):
                done += 1
                continue

            message = {"round": round_number, "agent": agent_id, "text": text}
            chat.append(message)
            save_message(path, message)

        if done == len(reviewer_ids):
            break

    return chat

def summarize(builder_id, chat):
    # the next builder turns the whole chat into short feedback for its own turn (no tools)
    messages = [{"role": "user", "content": SUMMARIZE.format(chat=format_chat(chat))}]
    return agent_ask(builder_id, messages)
