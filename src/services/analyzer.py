from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from src.core.types import RunLog
from src.llm.client import LLMClient
from src.agents.eda_agent import initial_eda_plan
from src.planning.loop import execute_plan, next_eda_plan
from src.reports.report import render_eda
from src.agents.model_agent import infer_task_and_target, train_candidates
from src.utils.config import AppConfig
from src.utils.s3 import upload_directory_to_s3


@dataclass
class AnalysisOptions:
    reflect_rounds: int = 2
    fast_test: bool = False
    no_multiproc: bool = False
    upload_s3: bool = False


def run_analysis(csv_path: str, out_dir: str = "data/artifacts", use_llm: bool = True, options: AnalysisOptions | None = None) -> dict:
    options = options or AnalysisOptions()
    log = RunLog(meta={"input_csv": csv_path})

    llm = None
    if use_llm:
        try:
            llm = LLMClient()
        except Exception:
            llm = None

    # EDA
    steps = initial_eda_plan(llm) if llm else []
    log = execute_plan(steps, csv_path, log)
    for _ in range(options.reflect_rounds):
        if not llm:
            break
        steps = next_eda_plan(llm, log)
        log = execute_plan(steps, csv_path, log)

    # Report
    render_eda(csv_path, artifacts_dir=out_dir)

    # Modeling
    df = pd.read_csv(csv_path)
    task, ycol = infer_task_and_target(df)
    scores = train_candidates(df, task, ycol)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "model_scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    s3_result: dict | None = None
    if options.upload_s3:
        cfg = AppConfig.load()
        if cfg.s3_bucket:
            try:
                uploaded = upload_directory_to_s3(out_dir, cfg.s3_bucket, cfg.s3_prefix)
                s3_result = {"bucket": cfg.s3_bucket, "prefix": cfg.s3_prefix, "uploaded": uploaded}
            except Exception as e:
                s3_result = {"error": str(e)}
        else:
            s3_result = {"error": "S3_BUCKET not set"}

    return {
        "report_md": str(Path(out_dir) / "eda_report.md"),
        "scores": scores,
        "turns": len(log.turns),
        "s3": s3_result,
    }


