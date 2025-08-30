# src/utils/dashboard.py
"""リアルタイム進捗ダッシュボード機能"""

from __future__ import annotations
import time
import json
import os
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False

from .progress import ProgressManager, ProgressState


class StatusDashboard:
    """リアルタイム状況表示ダッシュボード"""
    
    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        self.dashboard_data_path = Path("data/artifacts/dashboard.json")
        self.dashboard_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Streamlit用のコンテナ
        self.containers = {}
        self.last_update = 0.0
        self.update_interval = 0.5  # 秒
    
    def create_streamlit_dashboard(self) -> Dict[str, Any]:
        """Streamlitダッシュボードを作成"""
        if not HAS_STREAMLIT:
            return {}
        
        # メインコンテナ作成
        main_container = st.container()
        
        with main_container:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # 現在のフェーズ表示
                phase_container = st.empty()
                self.containers['phase'] = phase_container
            
            with col2:
                # 進捗率表示
                progress_container = st.empty()
                self.containers['progress'] = progress_container
            
            with col3:
                # ETA表示
                eta_container = st.empty()
                self.containers['eta'] = eta_container
            
            # 詳細進捗バー
            progress_bar = st.progress(0.0)
            self.containers['progress_bar'] = progress_bar
            
            # ステータス詳細
            status_detail = st.empty()
            self.containers['status_detail'] = status_detail
            
            # アーティファクト一覧
            with st.expander("生成されたアーティファクト", expanded=False):
                artifacts_container = st.empty()
                self.containers['artifacts'] = artifacts_container
            
            # フェーズ履歴
            with st.expander("フェーズ履歴", expanded=False):
                history_container = st.empty()
                self.containers['history'] = history_container
            
            # リアルタイムログ
            with st.expander("リアルタイムログ", expanded=True):
                log_container = st.empty()
                self.containers['log'] = log_container
        
        return self.containers
    
    def update_streamlit_dashboard(self):
        """Streamlitダッシュボードを更新"""
        if not HAS_STREAMLIT or not self.containers:
            return
        
        now = time.time()
        if now - self.last_update < self.update_interval:
            return
        
        state = self.progress_manager.state
        overall = self.progress_manager.get_overall_progress()
        
        # フェーズ表示更新
        if 'phase' in self.containers:
            status_icon = {"running": "🔄", "completed": "✅", "error": "❌", "paused": "⏸️"}.get(state.status, "")
            self.containers['phase'].markdown(f"### {status_icon} {state.phase}")
        
        # 進捗率表示更新
        if 'progress' in self.containers:
            progress_percent = int(state.progress * 100)
            self.containers['progress'].metric(
                "進捗率", 
                f"{progress_percent}%",
                delta=None
            )
        
        # ETA表示更新
        if 'eta' in self.containers:
            eta_text = "計算中..."
            if state.eta_sec is not None:
                if state.eta_sec < 60:
                    eta_text = f"{int(state.eta_sec)}秒"
                else:
                    eta_min = int(state.eta_sec / 60)
                    eta_sec = int(state.eta_sec % 60)
                    eta_text = f"{eta_min}分{eta_sec}秒"
            
            self.containers['eta'].metric("予想残り時間", eta_text)
        
        # プログレスバー更新
        if 'progress_bar' in self.containers:
            self.containers['progress_bar'].progress(state.progress)
        
        # ステータス詳細更新
        if 'status_detail' in self.containers:
            detail_data = {
                "現在のサブフェーズ": state.sub_phase or "なし",
                "ステップ": f"{state.current_step}/{state.total_steps}" if state.total_steps > 0 else f"{state.current_step}",
                "経過時間": f"{int(state.elapsed_sec)}秒",
                "ステータス": state.status,
                "メッセージ": state.message or "なし"
            }
            self.containers['status_detail'].json(detail_data)
        
        # アーティファクト一覧更新
        if 'artifacts' in self.containers:
            if state.artifacts_generated:
                artifacts_text = "\n".join([f"• {artifact}" for artifact in state.artifacts_generated])
                self.containers['artifacts'].markdown(artifacts_text)
            else:
                self.containers['artifacts'].info("まだアーティファクトは生成されていません")
        
        # フェーズ履歴更新
        if 'history' in self.containers:
            if self.progress_manager.phase_history:
                history_data = []
                for i, hist in enumerate(self.progress_manager.phase_history):
                    history_data.append({
                        "フェーズ": hist.phase,
                        "ステータス": hist.status,
                        "進捗率": f"{int(hist.progress * 100)}%",
                        "所要時間": f"{int(hist.elapsed_sec)}秒"
                    })
                
                import pandas as pd
                df = pd.DataFrame(history_data)
                self.containers['history'].dataframe(df)
            else:
                self.containers['history'].info("フェーズ履歴はまだありません")
        
        # リアルタイムログ更新
        if 'log' in self.containers:
            log_entries = self._get_recent_log_entries(10)
            if log_entries:
                log_text = "\n".join(log_entries)
                self.containers['log'].code(log_text, language="text")
            else:
                self.containers['log'].info("ログエントリはまだありません")
        
        self.last_update = now
    
    def create_console_dashboard(self) -> str:
        """コンソール用ダッシュボード文字列を作成"""
        state = self.progress_manager.state
        overall = self.progress_manager.get_overall_progress()
        
        # ヘッダー
        lines = ["=" * 60]
        lines.append("🚀 Data Scientist Agent - 進捗ダッシュボード")
        lines.append("=" * 60)
        
        # 現在のフェーズ
        status_icon = {"running": "🔄", "completed": "✅", "error": "❌", "paused": "⏸️"}.get(state.status, "")
        lines.append(f"現在のフェーズ: {status_icon} {state.phase}")
        
        if state.sub_phase:
            lines.append(f"サブフェーズ: {state.sub_phase}")
        
        # 進捗バー
        progress_width = 40
        filled = int(progress_width * state.progress)
        bar = "█" * filled + "░" * (progress_width - filled)
        progress_percent = int(state.progress * 100)
        lines.append(f"進捗: [{bar}] {progress_percent}%")
        
        # ステップ情報
        if state.total_steps > 0:
            lines.append(f"ステップ: {state.current_step}/{state.total_steps}")
        else:
            lines.append(f"ステップ: {state.current_step}")
        
        # 時間情報
        elapsed_min = int(state.elapsed_sec / 60)
        elapsed_sec = int(state.elapsed_sec % 60)
        lines.append(f"経過時間: {elapsed_min}分{elapsed_sec}秒")
        
        if state.eta_sec is not None:
            eta_min = int(state.eta_sec / 60)
            eta_sec = int(state.eta_sec % 60)
            lines.append(f"予想残り時間: {eta_min}分{eta_sec}秒")
        
        # メッセージ
        if state.message:
            lines.append(f"メッセージ: {state.message}")
        
        # アーティファクト
        if state.artifacts_generated:
            lines.append(f"生成済みアーティファクト: {len(state.artifacts_generated)}個")
            for artifact in state.artifacts_generated[-3:]:  # 最新3個のみ表示
                lines.append(f"  • {artifact}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_dashboard_snapshot(self):
        """ダッシュボードのスナップショットを保存"""
        try:
            state = self.progress_manager.state
            overall = self.progress_manager.get_overall_progress()
            
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_state": asdict(state),
                "overall_progress": overall,
                "console_dashboard": self.create_console_dashboard()
            }
            
            with open(self.dashboard_data_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"Failed to save dashboard snapshot: {e}")
    
    def _get_recent_log_entries(self, count: int = 10) -> List[str]:
        """最近のログエントリを取得"""
        log_file = Path("data/artifacts/progress.log")
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 最新のログエントリを取得
            recent_lines = lines[-count:] if len(lines) > count else lines
            
            # JSONログをパースして読みやすい形式に変換
            formatted_entries = []
            for line in recent_lines:
                try:
                    entry = json.loads(line.strip())
                    timestamp = entry.get('timestamp', '')[:19]  # 秒まで
                    phase = entry.get('phase', '')
                    message = entry.get('message', '')
                    progress = entry.get('progress', 0) * 100
                    
                    formatted_entries.append(f"[{timestamp}] {phase} ({progress:.1f}%) - {message}")
                except:
                    # JSONでない行はそのまま追加
                    formatted_entries.append(line.strip())
            
            return formatted_entries
        
        except Exception as e:
            return [f"ログ読み込みエラー: {e}"]


def create_dashboard_callback(dashboard: StatusDashboard) -> Callable:
    """ダッシュボード更新用コールバックを作成"""
    def dashboard_callback(state: ProgressState):
        # Streamlitダッシュボード更新
        dashboard.update_streamlit_dashboard()
        
        # スナップショット保存（5秒間隔）
        if not hasattr(dashboard_callback, '_last_snapshot'):
            dashboard_callback._last_snapshot = 0
        
        if time.time() - dashboard_callback._last_snapshot > 5.0:
            dashboard.save_dashboard_snapshot()
            dashboard_callback._last_snapshot = time.time()
    
    return dashboard_callback


def print_dashboard_summary(progress_manager: ProgressManager):
    """コンソールにダッシュボード要約を出力"""
    dashboard = StatusDashboard(progress_manager)
    print(dashboard.create_console_dashboard())
