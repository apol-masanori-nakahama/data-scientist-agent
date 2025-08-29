# src/planning/loop.py
from __future__ import annotations
import json
from typing import List, Optional, Callable
from src.core.types import Step, Observation, Turn, RunLog
from src.runners.code_runner import run_python
from src.llm.client import LLMClient
from src.agents.eda_agent import EDA_SYSTEM

SYSTEM = "You are a data analysis agent. Always return a JSON list of steps with action in {python,note,stop}."

def execute_plan(
    steps: List[Step],
    input_csv: str,
    log: RunLog,
    on_step: Optional[Callable[[int, Step, Observation], None]] = None,
) -> RunLog:
    for idx, s in enumerate(steps):
        if s.action == "python" and s.code:
            r = run_python(s.code, input_csv=input_csv)
            obs = Observation(r["ok"], r["stdout"], r["stderr"], r["artifacts"])
            log.turns.append(Turn(step=s, observation=obs))
            if on_step:
                on_step(idx, s, obs)
            if not r["ok"]:
                break
        elif s.action == "note":
            obs = Observation(True, s.note or "", "", [])
            log.turns.append(Turn(step=s, observation=obs))
            if on_step:
                on_step(idx, s, obs)
        elif s.action == "stop":
            obs = Observation(True, "stop", "", [])
            log.turns.append(Turn(step=s, observation=obs))
            if on_step:
                on_step(idx, s, obs)
            break
    return log

# src/planning/loop.py の next_plan を EDA反省版に差し替え
EDA_REFLECT = """Given the last observations, improve the EDA.
If errors occurred, choose simpler alternative (e.g., smaller plots, drop problematic columns).
Stop after you have: dtypes, missingness, describe, at least 3 figures.
Return JSON steps.
"""

def next_eda_plan(llm: LLMClient, log: RunLog) -> list[Step]:
    import json
    history = []
    for t in log.turns[-6:]:
        history.append({
            "action": t.step.action,
            "code": t.step.code[:400] if t.step.code else None,
            "ok": t.observation.success,
            "stdout": t.observation.stdout[-400:],
            "stderr": t.observation.stderr[-400:]
        })
    msg = [
        {"role":"system","content":EDA_SYSTEM},
        {"role":"user","content":f"History:\n```json\n{json.dumps(history,ensure_ascii=False)}```\n{EDA_REFLECT}"}
    ]
    content = llm.chat(msg)
    try:
        return [Step(**s) for s in json.loads(content)]
    except Exception:
        return [Step(action="stop", note="planning_error")]