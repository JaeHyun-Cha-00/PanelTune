import sys
import yaml
from paneltune.relay import run

# usage: uv run python -m scripts.run configs/panel.yaml
#        uv run python -m scripts.run configs/baseline.yaml 3     (solo run for agent 3)

# the guard matters: HumanEval starts helper processes that load this file again,
# and without it each helper would start a whole new run
if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    solo_agent = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(config, solo_agent)
