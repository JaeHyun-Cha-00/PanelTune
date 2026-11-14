# Panel

The four agents that build and review the student.

## Agents

| Agent | Model | Provider |
|---|---|---|
| 1 | `Qwen/Qwen3.5-4B` | DeepInfra |
| 2 | `google/gemma-4-E4B-it` | DeepInfra |
| 3 | `meta-llama/Llama-3.2-3B-Instruct` | DeepInfra |
| 4 | `mistralai/ministral-3b-2512` | OpenRouter |

All four are called through OpenAI-compatible APIs at temperature 0.7. Tokens used by each agent are recorded.

## Files

| File | What it does |
|---|---|
| `agents.py` | Calls the agents, and runs the tool loop (up to 5 rounds of tool calls per message) |
| `tools.py` | The tools: plain Python functions, plus the definitions the agents see |
| `prompts.py` | All prompt text (TBD) |
| `chat.py` | The group-chat discussion (TBD) |

## Tools

| Tool | What it does | Builder | Reviewers |
|---|---|---|---|
| `search_library` | Search the dataset library by meaning | Yes | |
| `list_files` / `read_file` | List and read files in the current run's folder | Yes | Yes |
| `write_note` | Save a note to the agent's own notebook, kept across turns | | Yes |
| `read_note` | Read the agent's own notebook | Yes | Yes |

No web search, so agents can't pull in benchmark test questions.
