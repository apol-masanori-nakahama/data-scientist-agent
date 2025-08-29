# src/reports/report.py
from __future__ import annotations
from jinja2 import Template
from pathlib import Path
import pandas as pd

MD_TMPL = Template("""
# EDA Report

**Input:** {{ csv_path }}

## Schema & Missingness
- Dtypes: see `cell_dtypes.csv`
- Missingness: see `cell_missing.csv`

## Raw Describe
{{ describe_text }}
""")

def render_eda(csv_path: str, artifacts_dir: str = "data/artifacts") -> str:
    p = Path(artifacts_dir)
    # Images are shown inline in the app; omit from Markdown to avoid broken links in some renderers
    desc = p / "cell_desc.csv"
    desc_text = ""
    if desc.exists():
        try:
            df = pd.read_csv(desc)
            desc_text = df.to_string()
        except Exception:
            desc_text = "(desc not readable)"
    md = MD_TMPL.render(csv_path=csv_path, describe_text=desc_text)
    out_md = p / "eda_report.md"
    out_md.write_text(md, encoding="utf-8")
    (p / "eda_report.html").write_text(f"<html><body><pre>{md}</pre></body></html>", encoding="utf-8")
    return str(out_md)
