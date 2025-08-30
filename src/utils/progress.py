# src/utils/progress.py
from __future__ import annotations
import time
import json
import os
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timezone


@dataclass
class ProgressState:
    """進捗状態を表すデータクラス"""
    phase: str
    sub_phase: str = ""
    current_step: int = 0
    total_steps: int = 0
    progress: float = 0.0
    elapsed_sec: float = 0.0
    eta_sec: Optional[float] = None
    start_time: float = 0.0
    last_update: float = 0.0
    status: str = "running"  # running, completed, error, paused
    message: str = ""
    artifacts_generated: List[str] = field(default_factory=list)


class ProgressManager:
    """包括的な進捗管理システム"""
    
    def __init__(self, save_path: str = "data/artifacts/progress.json"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.state = ProgressState(phase="初期化中", start_time=time.time())
        self.phase_history: List[ProgressState] = []
        self.callbacks: List[Callable[[ProgressState], None]] = []
        self.step_times: Dict[int, float] = {}
        
        # 自動保存設定
        self.auto_save = True
        self.save_interval = 1.0  # 秒
        self.last_save_time = 0.0
    
    def add_callback(self, callback: Callable[[ProgressState], None]):
        """進捗更新時のコールバックを追加"""
        self.callbacks.append(callback)
    
    def start_phase(self, phase_name: str, total_steps: int = 0, message: str = ""):
        """新しいフェーズを開始"""
        # 前フェーズを履歴に保存
        if self.state.phase != "初期化中":
            self.phase_history.append(self.state)
        
        # 新しいフェーズの状態を初期化
        self.state = ProgressState(
            phase=phase_name,
            total_steps=total_steps,
            start_time=time.time(),
            last_update=time.time(),
            message=message,
            artifacts_generated=[]
        )
        self.step_times = {}
        self._notify_and_save()
    
    def update_step(self, 
                   step_name: str = "", 
                   increment: int = 1, 
                   message: str = "",
                   artifacts: Optional[List[str]] = None):
        """ステップ進捗を更新"""
        now = time.time()
        
        # ステップ数更新
        self.state.current_step += increment
        self.state.sub_phase = step_name
        self.state.message = message or self.state.message
        
        # 時間記録
        self.step_times[self.state.current_step] = now
        self.state.elapsed_sec = now - self.state.start_time
        self.state.last_update = now
        
        # 進捗率計算
        if self.state.total_steps > 0:
            self.state.progress = min(1.0, self.state.current_step / self.state.total_steps)
        
        # ETA計算
        self._calculate_eta()
        
        # アーティファクト追加
        if artifacts:
            self.state.artifacts_generated.extend(artifacts)
        
        self._notify_and_save()
    
    def set_total_steps(self, total: int):
        """総ステップ数を設定（動的に変更可能）"""
        self.state.total_steps = total
        if total > 0:
            self.state.progress = min(1.0, self.state.current_step / total)
        self._notify_and_save()
    
    def complete_phase(self, message: str = "完了"):
        """現在のフェーズを完了"""
        self.state.status = "completed"
        self.state.progress = 1.0
        self.state.message = message
        self.state.last_update = time.time()
        self._notify_and_save()
    
    def error_phase(self, error_message: str):
        """エラー状態に設定"""
        self.state.status = "error"
        self.state.message = error_message
        self.state.last_update = time.time()
        self._notify_and_save()
    
    def pause_phase(self, message: str = "一時停止"):
        """フェーズを一時停止"""
        self.state.status = "paused"
        self.state.message = message
        self.state.last_update = time.time()
        self._notify_and_save()
    
    def resume_phase(self, message: str = "再開"):
        """フェーズを再開"""
        self.state.status = "running"
        self.state.message = message
        self.state.last_update = time.time()
        self._notify_and_save()
    
    def _calculate_eta(self):
        """ETA（推定残り時間）を計算"""
        if self.state.current_step <= 0 or self.state.total_steps <= 0:
            self.state.eta_sec = None
            return
        
        if self.state.progress >= 1.0:
            self.state.eta_sec = 0.0
            return
        
        # 平均処理時間を計算
        if len(self.step_times) >= 2:
            times = list(self.step_times.values())
            intervals = [times[i] - times[i-1] for i in range(1, len(times))]
            avg_step_time = sum(intervals) / len(intervals)
        else:
            avg_step_time = self.state.elapsed_sec / max(1, self.state.current_step)
        
        remaining_steps = self.state.total_steps - self.state.current_step
        self.state.eta_sec = avg_step_time * remaining_steps
    
    def _notify_and_save(self):
        """コールバック実行と自動保存"""
        # コールバック実行
        for callback in self.callbacks:
            try:
                callback(self.state)
            except Exception as e:
                print(f"Progress callback error: {e}")
        
        # 自動保存
        if self.auto_save and (time.time() - self.last_save_time) >= self.save_interval:
            self.save_state()
    
    def save_state(self):
        """進捗状態をファイルに保存"""
        try:
            data = {
                "current_state": asdict(self.state),
                "phase_history": [asdict(state) for state in self.phase_history],
                "total_phases": len(self.phase_history) + 1,
                "overall_start_time": self.phase_history[0].start_time if self.phase_history else self.state.start_time,
                "last_saved": datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.last_save_time = time.time()
        except Exception as e:
            print(f"Failed to save progress state: {e}")
    
    def load_state(self) -> bool:
        """保存された進捗状態を読み込み"""
        try:
            if not self.save_path.exists():
                return False
            
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 現在の状態を復元
            state_dict = data["current_state"]
            self.state = ProgressState(**state_dict)
            
            # 履歴を復元
            self.phase_history = [ProgressState(**hist) for hist in data["phase_history"]]
            
            return True
        except Exception as e:
            print(f"Failed to load progress state: {e}")
            return False
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """全体の進捗情報を取得"""
        total_phases = len(self.phase_history) + 1
        completed_phases = len([h for h in self.phase_history if h.status == "completed"])
        
        if self.state.status == "completed":
            completed_phases += 1
        
        overall_progress = completed_phases / max(1, total_phases)
        if self.state.status == "running" and self.state.progress > 0:
            overall_progress += (1.0 / total_phases) * self.state.progress
        
        return {
            "overall_progress": min(1.0, overall_progress),
            "current_phase": self.state.phase,
            "total_phases": total_phases,
            "completed_phases": completed_phases,
            "current_state": asdict(self.state),
            "phase_history": [asdict(h) for h in self.phase_history]
        }
    
    def get_status_summary(self) -> str:
        """現在の状況の要約を取得"""
        eta_str = ""
        if self.state.eta_sec:
            eta_min = int(self.state.eta_sec / 60)
            eta_sec = int(self.state.eta_sec % 60)
            eta_str = f" (ETA: {eta_min}m{eta_sec}s)"
        
        progress_percent = int(self.state.progress * 100)
        
        return f"{self.state.phase} - {progress_percent}%{eta_str}"


def create_console_callback() -> Callable[[ProgressState], None]:
    """コンソール出力用のコールバックを作成"""
    def console_callback(state: ProgressState):
        eta_str = ""
        if state.eta_sec:
            eta_min = int(state.eta_sec / 60)
            eta_sec = int(state.eta_sec % 60)
            eta_str = f" (ETA: {eta_min}m{eta_sec}s)"
        
        progress_percent = int(state.progress * 100)
        step_info = f" [{state.current_step}/{state.total_steps}]" if state.total_steps > 0 else ""
        
        status_icon = {"running": "🔄", "completed": "✅", "error": "❌", "paused": "⏸️"}.get(state.status, "")
        
        print(f"{status_icon} {state.phase}{step_info} - {progress_percent}%{eta_str}")
        if state.sub_phase:
            print(f"  └─ {state.sub_phase}")
        if state.message:
            print(f"  💬 {state.message}")
    
    return console_callback


def create_file_callback(file_path: str) -> Callable[[ProgressState], None]:
    """ファイル出力用のコールバックを作成"""
    def file_callback(state: ProgressState):
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": state.phase,
                "sub_phase": state.sub_phase,
                "progress": state.progress,
                "status": state.status,
                "message": state.message,
                "eta_sec": state.eta_sec
            }
            
            # ログファイルに追記
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"File callback error: {e}")
    
    return file_callback
