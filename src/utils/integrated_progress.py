# src/utils/integrated_progress.py
"""統合進捗システム - すべての進捗機能を統合"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from .progress import ProgressManager, create_console_callback, create_file_callback
from .dashboard import StatusDashboard, create_dashboard_callback
from .notifications import NotificationManager, create_notification_callback, setup_notifications_from_env
from .checkpoint import CheckpointManager, setup_auto_checkpoint
from .logging import setup_logging, create_progress_logger


class IntegratedProgressSystem:
    """統合進捗システム - 全ての進捗機能を一括管理"""
    
    def __init__(self, 
                 enable_dashboard: bool = True,
                 enable_notifications: bool = True,
                 enable_checkpoints: bool = True,
                 enable_console_output: bool = True,
                 checkpoint_interval_minutes: int = 5):
        
        # コア進捗管理
        self.progress_manager = ProgressManager()
        
        # ダッシュボード
        self.dashboard: Optional[StatusDashboard] = None
        if enable_dashboard:
            self.dashboard = StatusDashboard(self.progress_manager)
            self.progress_manager.add_callback(create_dashboard_callback(self.dashboard))
        
        # 通知システム
        self.notification_manager: Optional[NotificationManager] = None
        if enable_notifications:
            self.notification_manager = setup_notifications_from_env()
            self.progress_manager.add_callback(create_notification_callback(self.notification_manager))
        
        # チェックポイント
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if enable_checkpoints:
            self.checkpoint_manager = setup_auto_checkpoint(
                self.progress_manager, 
                checkpoint_interval_minutes
            )
        
        # コンソール出力
        if enable_console_output:
            self.progress_manager.add_callback(create_console_callback())
        
        # ファイルログ
        log_path = "data/artifacts/progress.log"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.progress_manager.add_callback(create_file_callback(log_path))
        
        # ロギングセットアップ
        setup_logging()
        self.logger = create_progress_logger("integrated_progress")
    
    def start_phase(self, phase_name: str, total_steps: int = 0, message: str = ""):
        """フェーズ開始"""
        self.progress_manager.start_phase(phase_name, total_steps, message)
        
        # 通知送信
        if self.notification_manager:
            from .notifications import NotificationType
            self.notification_manager.send_notification(
                NotificationType.PHASE_START,
                f"フェーズ開始: {phase_name}",
                message or f"フェーズ '{phase_name}' を開始しました",
                self.progress_manager.state
            )
    
    def update_step(self, step_name: str = "", message: str = "", artifacts: Optional[List[str]] = None):
        """ステップ更新"""
        self.progress_manager.update_step(step_name, message=message, artifacts=artifacts or [])
    
    def complete_phase(self, message: str = "完了"):
        """フェーズ完了"""
        self.progress_manager.complete_phase(message)
    
    def error_phase(self, error_message: str):
        """エラー状態"""
        self.progress_manager.error_phase(error_message)
    
    def get_streamlit_containers(self) -> Dict[str, Any]:
        """Streamlitコンテナを取得"""
        if self.dashboard:
            return self.dashboard.create_streamlit_dashboard()
        return {}
    
    def get_status_summary(self) -> str:
        """現在の状況要約を取得"""
        return self.progress_manager.get_status_summary()
    
    def save_checkpoint(self, description: str = "") -> str:
        """手動チェックポイント作成"""
        if self.checkpoint_manager:
            return self.checkpoint_manager.create_checkpoint(
                self.progress_manager, 
                description, 
                force=True
            )
        return ""
    
    def list_checkpoints(self):
        """チェックポイント一覧"""
        if self.checkpoint_manager:
            return self.checkpoint_manager.list_checkpoints()
        return []
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """チェックポイント復元"""
        if self.checkpoint_manager:
            restored_manager = self.checkpoint_manager.restore_checkpoint(checkpoint_id)
            if restored_manager:
                # 進捗管理を復元されたものに置き換え
                self.progress_manager = restored_manager
                
                # コールバックを再設定
                if self.dashboard:
                    self.dashboard.progress_manager = self.progress_manager
                    self.progress_manager.add_callback(create_dashboard_callback(self.dashboard))
                
                if self.notification_manager:
                    self.progress_manager.add_callback(create_notification_callback(self.notification_manager))
                
                return True
        return False
    
    def get_overall_status(self) -> Dict[str, Any]:
        """全体ステータス取得"""
        status = {
            "progress": self.progress_manager.get_overall_progress(),
            "current_phase": self.progress_manager.state.phase,
            "status": self.progress_manager.state.status,
            "message": self.progress_manager.state.message
        }
        
        if self.checkpoint_manager:
            status["checkpoint_info"] = self.checkpoint_manager.get_checkpoint_size()
        
        return status


def create_integrated_progress_system(
    enable_all: bool = True,
    enable_dashboard: bool = None,
    enable_notifications: bool = None,
    enable_checkpoints: bool = None,
    enable_console_output: bool = None
) -> IntegratedProgressSystem:
    """統合進捗システムを作成（環境変数対応）"""
    
    # 環境変数から設定を読み込み
    if enable_dashboard is None:
        enable_dashboard = enable_all and os.getenv("PROGRESS_DASHBOARD", "true").lower() == "true"
    
    if enable_notifications is None:
        enable_notifications = enable_all and os.getenv("PROGRESS_NOTIFICATIONS", "true").lower() == "true"
    
    if enable_checkpoints is None:
        enable_checkpoints = enable_all and os.getenv("PROGRESS_CHECKPOINTS", "true").lower() == "true"
    
    if enable_console_output is None:
        enable_console_output = enable_all and os.getenv("PROGRESS_CONSOLE", "true").lower() == "true"
    
    checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL_MINUTES", "5"))
    
    return IntegratedProgressSystem(
        enable_dashboard=enable_dashboard,
        enable_notifications=enable_notifications,
        enable_checkpoints=enable_checkpoints,
        enable_console_output=enable_console_output,
        checkpoint_interval_minutes=checkpoint_interval
    )


# グローバル進捗システムインスタンス（シングルトン的な使用）
_global_progress_system: Optional[IntegratedProgressSystem] = None


def get_global_progress_system() -> IntegratedProgressSystem:
    """グローバル進捗システムを取得"""
    global _global_progress_system
    if _global_progress_system is None:
        _global_progress_system = create_integrated_progress_system()
    return _global_progress_system


def reset_global_progress_system():
    """グローバル進捗システムをリセット"""
    global _global_progress_system
    _global_progress_system = None
