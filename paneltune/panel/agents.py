import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from paneltune.panel.tools import TOOL_FUNCTIONS, NEEDS_AGENT_ID, REVIEWER_TOOLS

load_dotenv()

deepinfra_client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai",
    max_retries=3,
)

openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_retries=3,
)

agent_dict = {
    1: {"name": "Qwen3.5 4B", "model": "Qwen/Qwen3.5-4B", "provider": "deepinfra"},
    2: {"name": "Gemma 4 E4B", "model": "google/gemma-4-E4B-it", "provider": "deepinfra"},
    3: {"name": "Llama 3.2 3B", "model": "meta-llama/Llama-3.2-3B-Instruct", "provider": "deepinfra"},
    4: {"name": "Granite 4.2 3B", "model": "ibm-granite/granite-4.2-3b", "provider": "deepinfra"},
    5: {"name": "Ministral 3B", "model": "mistralai/ministral-3b-2512", "provider": "openrouter"},
}

token_usage = {
    1: {"prompt": 0, "completion": 0},
    2: {"prompt": 0, "completion": 0},
    3: {"prompt": 0, "completion": 0},
    4: {"prompt": 0, "completion": 0},
    5: {"prompt": 0, "completion": 0},
}

def run_tool(agent_id, call):

    name = call.function.name
    args = json.loads(call.function.arguments or "{}")

    if name in NEEDS_AGENT_ID:
        args["agent_id"] = agent_id

    return TOOL_FUNCTIONS[name](**args)

def agent_ask(agent_id, messages, tools=None):

    agent = agent_dict[agent_id]
    messages = list(messages)

    if agent["provider"] == "deepinfra":
        client = deepinfra_client
    else:
        client = openrouter_client

    for _ in range(5):
        response = client.chat.completions.create(
            model=agent["model"],
            messages=messages,
            temperature=0.7,
            tools=tools,
        )

        token_usage[agent_id]["prompt"] += response.usage.prompt_tokens
        token_usage[agent_id]["completion"] += response.usage.completion_tokens

        message = response.choices[0].message

        if not message.tool_calls:
            if message.content is None:
                return ""
            return message.content.strip()

        messages.append(message.model_dump(exclude_none=True))

        for call in message.tool_calls:
            result = run_tool(agent_id, call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "(stopped: too many tool calls)"

if __name__ == "__main__":
    for agent_id in agent_dict:
        name = agent_dict[agent_id]["name"]
        messages = [
            {"role": "user", "content": "Save a note saying 'test note', then read your notes and tell me what they say."},
        ]
        reply = agent_ask(agent_id, messages, tools=REVIEWER_TOOLS)
        print(f"{name}: {reply}")

    print(token_usage)
