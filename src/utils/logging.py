from __future__ import annotations
import json, logging, sys, time, uuid
from typing import Any, Optional, Dict
from datetime import datetime, timezone


class ProgressTracker:
    """進捗追跡とETA計算のためのヘルパークラス"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.current_step: int = 0
        self.total_steps: int = 0
        self.step_times: Dict[int, float] = {}
        self.phase_name: str = ""
        self.sub_phase: str = ""
    
    def start_phase(self, phase_name: str, total_steps: int):
        """フェーズ開始"""
        self.phase_name = phase_name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = {}
    
    def step_completed(self, step_name: str = ""):
        """ステップ完了時の進捗更新"""
        now = time.time()
        self.current_step += 1
        self.step_times[self.current_step] = now
        self.sub_phase = step_name
    
    def get_progress_info(self) -> Dict[str, Any]:
        """現在の進捗情報を取得"""
        if not self.start_time:
            return {"phase": self.phase_name, "progress": 0.0}
        
        now = time.time()
        elapsed = now - self.start_time
        progress = self.current_step / max(1, self.total_steps)
        
        # ETA計算
        eta_seconds = None
        if self.current_step > 0 and progress < 1.0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_time_per_step * remaining_steps
        
        return {
            "phase": self.phase_name,
            "sub_phase": self.sub_phase,
            "step": f"{self.current_step}/{self.total_steps}",
            "progress": min(1.0, progress),
            "elapsed_sec": elapsed,
            "eta_sec": eta_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        
        # 進捗関連の情報を追加
        for k in ("request_id", "path", "status", "elapsed_ms", "progress", "phase", 
                 "sub_phase", "step", "eta_sec", "artifacts_count"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def new_request_id() -> str:
    return uuid.uuid4().hex


def create_progress_logger(name: str = "progress") -> logging.Logger:
    """進捗専用のロガーを作成"""
    logger = logging.getLogger(name)
    return logger


def log_progress(logger: logging.Logger, tracker: ProgressTracker, message: str = "", **extra_fields):
    """進捗情報をログに記録"""
    progress_info = tracker.get_progress_info()
    
    # ログレコードに進捗情報を追加
    extra = {**progress_info, **extra_fields}
    
    # メッセージが空の場合はデフォルトメッセージを生成
    if not message:
        if progress_info.get("eta_sec"):
            eta_min = int(progress_info["eta_sec"] / 60)
            eta_sec = int(progress_info["eta_sec"] % 60)
            message = f"{progress_info['phase']} - {progress_info['step']} (ETA: {eta_min}m{eta_sec}s)"
        else:
            message = f"{progress_info['phase']} - {progress_info['step']}"
    
    logger.info(message, extra=extra)


