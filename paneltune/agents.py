import os
from dotenv import load_dotenv
from openai import OpenAI

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

def agent_ask(agent_id, messages):

    agent = agent_dict[agent_id]

    if agent["provider"] == "deepinfra":
        client = deepinfra_client
    else:
        client = openrouter_client

    response = client.chat.completions.create(
        model=agent["model"],
        messages=messages,
        temperature=0.7
    )

    token_usage[agent_id]["prompt"] += response.usage.prompt_tokens
    token_usage[agent_id]["completion"] += response.usage.completion_tokens

    content = response.choices[0].message.content

    if content is None:
        return ""

    return content.strip()

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi in one sentence."},
    ]

    for agent_id in agent_dict:
        name = agent_dict[agent_id]["name"]
        try:
            reply = agent_ask(agent_id, messages)
            print(f"{name}: {reply}")
        except Exception as e:
            print(f"{name}: ERROR {e}")
    
    print(token_usage)