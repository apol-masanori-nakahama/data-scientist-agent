# src/utils/notifications.py
"""進捗通知システム"""

from __future__ import annotations
import os
import json
import time
import smtplib
from typing import Dict, Any, Optional, List, Callable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from .progress import ProgressState


class NotificationType(Enum):
    """通知タイプ"""
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    PHASE_ERROR = "phase_error"
    MILESTONE = "milestone"
    ETA_UPDATE = "eta_update"
    COMPLETION = "completion"


@dataclass
class NotificationConfig:
    """通知設定"""
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_from: str = ""
    email_to: Optional[List[str]] = None
    email_password: str = ""
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    
    desktop_enabled: bool = False
    sound_enabled: bool = False
    
    file_log_enabled: bool = True
    file_log_path: str = "data/artifacts/notifications.log"
    
    # 通知頻度制御
    min_interval_seconds: float = 30.0  # 最小通知間隔
    milestone_progress_steps: Optional[List[float]] = None  # マイルストーン進捗率
    
    def __post_init__(self):
        if self.email_to is None:
            self.email_to = []
        if self.milestone_progress_steps is None:
            self.milestone_progress_steps = [0.25, 0.5, 0.75, 1.0]


class NotificationManager:
    """通知管理システム"""
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self.last_notification_time: Dict[NotificationType, float] = {}
        self.milestone_sent: Dict[float, bool] = {}
        
        # 環境変数から設定を読み込み
        self._load_config_from_env()
        
        # ログファイル準備
        if self.config.file_log_enabled:
            log_path = Path(self.config.file_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_config_from_env(self):
        """環境変数から設定を読み込み"""
        # メール設定
        if os.getenv("NOTIFICATION_EMAIL_ENABLED", "false").lower() == "true":
            self.config.email_enabled = True
            self.config.email_from = os.getenv("NOTIFICATION_EMAIL_FROM", "")
            self.config.email_to = os.getenv("NOTIFICATION_EMAIL_TO", "").split(",")
            self.config.email_password = os.getenv("NOTIFICATION_EMAIL_PASSWORD", "")
            self.config.email_smtp_server = os.getenv("NOTIFICATION_EMAIL_SMTP", "smtp.gmail.com")
            self.config.email_smtp_port = int(os.getenv("NOTIFICATION_EMAIL_PORT", "587"))
        
        # Webhook設定
        if os.getenv("NOTIFICATION_WEBHOOK_ENABLED", "false").lower() == "true":
            self.config.webhook_enabled = True
            self.config.webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL", "")
        
        # デスクトップ通知設定
        if os.getenv("NOTIFICATION_DESKTOP_ENABLED", "false").lower() == "true":
            self.config.desktop_enabled = True
        
        # サウンド通知設定
        if os.getenv("NOTIFICATION_SOUND_ENABLED", "false").lower() == "true":
            self.config.sound_enabled = True
    
    def should_send_notification(self, notification_type: NotificationType) -> bool:
        """通知を送信すべきかチェック"""
        now = time.time()
        last_time = self.last_notification_time.get(notification_type, 0)
        
        return (now - last_time) >= self.config.min_interval_seconds
    
    def send_notification(self, 
                         notification_type: NotificationType, 
                         title: str, 
                         message: str, 
                         state: Optional[ProgressState] = None,
                         force: bool = False):
        """通知を送信"""
        if not force and not self.should_send_notification(notification_type):
            return
        
        # 通知データ準備
        notification_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": notification_type.value,
            "title": title,
            "message": message,
            "state": state.__dict__ if state else None
        }
        
        # 各通知方法で送信
        if self.config.email_enabled:
            self._send_email_notification(notification_data)
        
        if self.config.webhook_enabled:
            self._send_webhook_notification(notification_data)
        
        if self.config.desktop_enabled:
            self._send_desktop_notification(notification_data)
        
        if self.config.sound_enabled:
            self._play_notification_sound(notification_type)
        
        if self.config.file_log_enabled:
            self._log_notification(notification_data)
        
        # 最後の通知時刻を更新
        self.last_notification_time[notification_type] = time.time()
    
    def _send_email_notification(self, notification_data: Dict[str, Any]):
        """メール通知を送信"""
        if not self.config.email_from or not self.config.email_to:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ", ".join(self.config.email_to)
            msg['Subject'] = f"[Data Scientist Agent] {notification_data['title']}"
            
            # メール本文作成
            body_lines = [
                f"タイトル: {notification_data['title']}",
                f"メッセージ: {notification_data['message']}",
                f"時刻: {notification_data['timestamp']}",
                f"通知タイプ: {notification_data['type']}"
            ]
            
            if notification_data['state']:
                state = notification_data['state']
                body_lines.extend([
                    "",
                    "=== 詳細情報 ===",
                    f"フェーズ: {state.get('phase', 'N/A')}",
                    f"進捗率: {int(state.get('progress', 0) * 100)}%",
                    f"ステータス: {state.get('status', 'N/A')}",
                    f"経過時間: {int(state.get('elapsed_sec', 0))}秒"
                ])
                
                if state.get('eta_sec'):
                    eta_min = int(state['eta_sec'] / 60)
                    eta_sec = int(state['eta_sec'] % 60)
                    body_lines.append(f"予想残り時間: {eta_min}分{eta_sec}秒")
            
            body = "\n".join(body_lines)
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # SMTP送信
            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                server.login(self.config.email_from, self.config.email_password)
                server.send_message(msg)
            
            print(f"✉️ メール通知送信完了: {notification_data['title']}")
        
        except Exception as e:
            print(f"❌ メール通知送信失敗: {e}")
    
    def _send_webhook_notification(self, notification_data: Dict[str, Any]):
        """Webhook通知を送信"""
        try:
            import requests
            
            payload = {
                "text": f"🤖 {notification_data['title']}\n{notification_data['message']}",
                "data": notification_data
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            print(f"🔗 Webhook通知送信完了: {notification_data['title']}")
        
        except Exception as e:
            print(f"❌ Webhook通知送信失敗: {e}")
    
    def _send_desktop_notification(self, notification_data: Dict[str, Any]):
        """デスクトップ通知を送信"""
        try:
            # macOS
            if os.system("which osascript > /dev/null 2>&1") == 0:
                title = notification_data['title']
                message = notification_data['message']
                os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')
                print(f"🖥️ デスクトップ通知送信完了: {title}")
                return
            
            # Linux (notify-send)
            if os.system("which notify-send > /dev/null 2>&1") == 0:
                title = notification_data['title']
                message = notification_data['message']
                os.system(f'notify-send "{title}" "{message}"')
                print(f"🖥️ デスクトップ通知送信完了: {title}")
                return
            
            # Windows (PowerShell)
            if os.name == 'nt':
                title = notification_data['title']
                message = notification_data['message']
                powershell_cmd = f'''
                Add-Type -AssemblyName System.Windows.Forms
                $notification = New-Object System.Windows.Forms.NotifyIcon
                $notification.Icon = [System.Drawing.SystemIcons]::Information
                $notification.BalloonTipTitle = "{title}"
                $notification.BalloonTipText = "{message}"
                $notification.Visible = $true
                $notification.ShowBalloonTip(5000)
                '''
                os.system(f'powershell -Command "{powershell_cmd}"')
                print(f"🖥️ デスクトップ通知送信完了: {title}")
                return
            
            print("⚠️ デスクトップ通知: サポートされていないプラットフォーム")
        
        except Exception as e:
            print(f"❌ デスクトップ通知送信失敗: {e}")
    
    def _play_notification_sound(self, notification_type: NotificationType):
        """通知音を再生"""
        try:
            # macOS
            if os.system("which afplay > /dev/null 2>&1") == 0:
                if notification_type == NotificationType.COMPLETION:
                    os.system("afplay /System/Library/Sounds/Glass.aiff")
                elif notification_type == NotificationType.PHASE_ERROR:
                    os.system("afplay /System/Library/Sounds/Sosumi.aiff")
                else:
                    os.system("afplay /System/Library/Sounds/Ping.aiff")
                return
            
            # Linux (aplay)
            if os.system("which aplay > /dev/null 2>&1") == 0:
                # 簡単なビープ音
                os.system("echo -e '\a'")
                return
            
            # Windows
            if os.name == 'nt':
                import winsound
                if notification_type == NotificationType.COMPLETION:
                    winsound.MessageBeep(winsound.MB_OK)
                elif notification_type == NotificationType.PHASE_ERROR:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                else:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                return
        
        except Exception as e:
            print(f"❌ 通知音再生失敗: {e}")
    
    def _log_notification(self, notification_data: Dict[str, Any]):
        """通知をファイルに記録"""
        try:
            with open(self.config.file_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(notification_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ 通知ログ記録失敗: {e}")
    
    def check_milestones(self, state: ProgressState):
        """マイルストーン進捗をチェックして通知"""
        for milestone in self.config.milestone_progress_steps:
            if (state.progress >= milestone and 
                not self.milestone_sent.get(milestone, False) and
                state.status == "running"):
                
                self.send_notification(
                    NotificationType.MILESTONE,
                    f"マイルストーン達成: {int(milestone * 100)}%",
                    f"フェーズ '{state.phase}' が {int(milestone * 100)}% 完了しました",
                    state
                )
                self.milestone_sent[milestone] = True


def create_notification_callback(notification_manager: NotificationManager) -> Callable[[ProgressState], None]:
    """通知用コールバックを作成"""
    def notification_callback(state: ProgressState):
        # マイルストーンチェック
        notification_manager.check_milestones(state)
        
        # フェーズ完了通知
        if state.status == "completed":
            notification_manager.send_notification(
                NotificationType.PHASE_COMPLETE,
                f"フェーズ完了: {state.phase}",
                f"フェーズ '{state.phase}' が正常に完了しました",
                state
            )
        
        # エラー通知
        elif state.status == "error":
            notification_manager.send_notification(
                NotificationType.PHASE_ERROR,
                f"フェーズエラー: {state.phase}",
                f"フェーズ '{state.phase}' でエラーが発生しました: {state.message}",
                state,
                force=True  # エラーは常に通知
            )
    
    return notification_callback


def setup_notifications_from_env() -> NotificationManager:
    """環境変数から通知設定をセットアップ"""
    config = NotificationConfig()
    return NotificationManager(config)
