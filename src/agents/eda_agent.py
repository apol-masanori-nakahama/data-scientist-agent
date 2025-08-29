# src/agents/eda_agent.py
from __future__ import annotations
import json
from typing import List, cast
from src.core.types import Step, RunLog, ActionType
from src.llm.client import LLMClient

EDA_SYSTEM = """You are an autonomous EDA agent.
Rules:
- Always return a JSON list of steps: [{"action":"python"|"note"|"stop","code"?:str,"note"?:str}]
- Prefer SAFE python using pandas/numpy/matplotlib only.
- Use INPUT_CSV variable as the data path.
- Save figures/tables under data/artifacts/.
- Cover: dtypes, missingness, basic stats, target inference (if any), correlations, distributions, time patterns.
- Keep each code cell focused and idempotent.
"""

EDA_SEED = """
Create 4-6 steps to: read CSV, infer dtypes, print shape & head, summarize missing values by column,
save describe(include='all'), plot top-5 numeric histograms, and output a short textual note with findings.
"""

def initial_eda_plan(llm: LLMClient) -> List[Step]:
    msg = [{"role":"system","content":EDA_SYSTEM},{"role":"user","content":EDA_SEED}]
    try:
        content = llm.chat(msg)
        arr = json.loads(content)
        return [Step(**s) for s in arr]
    except Exception:
        # Fallback to few-shot static steps when LLM is unavailable or returns invalid JSON
        fallback: List[Step] = []
        for s in EDA_FEWSHOT:
            fallback.append(
                Step(
                    action=cast(ActionType, s.get("action")),
                    code=s.get("code"),
                    note=s.get("note")
                )
            )
        return fallback

EDA_FEWSHOT = [
    {
      "action": "python",
      "code": """
import pandas as pd
df = pd.read_csv(INPUT_CSV)
print("shape:", df.shape)
print(df.head(5).to_string())
dtypes = df.dtypes.astype(str).reset_index()
dtypes.columns = ["column","dtype"]
dtypes.to_csv("data/artifacts/cell_dtypes.csv", index=False)
"""
    },
    {
      "action": "python",
      "code": """
import pandas as pd
df = pd.read_csv(INPUT_CSV)
miss = df.isna().sum().reset_index()
miss.columns = ["column","na_count"]
miss.to_csv("data/artifacts/cell_missing.csv", index=False)
"""
    },
    {
      "action": "python",
      "code": """
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(INPUT_CSV)
num = df.select_dtypes(include="number").columns[:5]
for c in num:
    ax = df[c].dropna().hist(bins=30)
    ax.figure.suptitle(f"hist_{c}")
    ax.figure.savefig(f"data/artifacts/cell_hist_{c}.png")
    plt.close(ax.figure)
"""
    },
    {
      "action": "python",
      "code": """
import pandas as pd
df = pd.read_csv(INPUT_CSV)
desc = df.describe(include='all').transpose()
desc.to_csv("data/artifacts/cell_desc.csv")
"""
    },
    { "action": "note", "note": "Saved dtypes, missingness, and histograms for up to 5 numeric columns." }
]