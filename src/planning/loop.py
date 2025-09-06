# src/planning/loop.py
from __future__ import annotations
import json
from typing import List, Optional, Callable
from src.core.types import Step, Observation, Turn, RunLog
from src.runners.code_runner import run_python
from src.llm.client import LLMClient
from src.agents.eda_agent import EDA_SYSTEM
from src.utils.progress import ProgressManager
from src.utils.logging import ProgressTracker, create_progress_logger, log_progress

SYSTEM = "You are a data analysis agent. Always return a JSON list of steps with action in {python,note,stop}."

def execute_plan(
    steps: List[Step],
    input_csv: str,
    log: RunLog,
    on_step: Optional[Callable[[int, Step, Observation], None]] = None,
    progress_manager: Optional[ProgressManager] = None,
    phase_name: str = "実行中",
) -> RunLog:
    # 進捗管理の初期化
    if progress_manager:
        # 既に同一フェーズ名で開始されている場合は、フェーズを再初期化せず総ステップのみ設定
        try:
            if getattr(progress_manager, "state", None) and progress_manager.state.phase == phase_name:
                progress_manager.set_total_steps(len(steps))
            else:
                progress_manager.start_phase(phase_name, len(steps))
        except Exception:
            # フェーズ状態取得に失敗した場合はフォールバックで開始
            progress_manager.start_phase(phase_name, len(steps))
    
    progress_logger = create_progress_logger("execute_plan")
    tracker = ProgressTracker()
    tracker.start_phase(phase_name, len(steps))
    
    for idx, s in enumerate(steps):
        # ステップ開始の進捗更新
        step_name = f"{s.action}: {(s.note or s.code or '')[:50]}..."
        
        if progress_manager:
            progress_manager.update_step(step_name)
        
        tracker.step_completed(step_name)
        log_progress(progress_logger, tracker, f"ステップ {idx + 1}/{len(steps)} 実行中")
        
        if s.action == "python" and s.code:
            r = run_python(s.code, input_csv=input_csv)
            obs = Observation(r["ok"], r["stdout"], r["stderr"], r["artifacts"])
            log.turns.append(Turn(step=s, observation=obs))
            
            # アーティファクト情報を進捗に追加（ステップ数は増やさない）
            if progress_manager and r["artifacts"]:
                progress_manager.update_step(increment=0, artifacts=r["artifacts"])
            
            if on_step:
                on_step(idx, s, obs)
            if not r["ok"]:
                if progress_manager:
                    progress_manager.error_phase(f"ステップ {idx + 1} でエラー: {r['stderr'][:100]}")
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
    
    # フェーズ完了
    if progress_manager:
        progress_manager.complete_phase(f"{phase_name}完了")
    
    return log

# src/planning/loop.py の next_plan を EDA反省版に差し替え
EDA_REFLECT = """Given the last observations, improve the EDA.
If errors occurred, choose simpler alternative (e.g., smaller plots, drop problematic columns).
Prefer one artifact per step; split multi-plot loops into separate steps for better progress visibility.
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
