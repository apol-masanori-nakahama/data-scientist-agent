# src/utils/checkpoint.py
"""チェックポイントシステム - 中間状態保存・復元機能"""

from __future__ import annotations
import json
import pickle
import os
import shutil
import sys
import time
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import hashlib

from .progress import ProgressState, ProgressManager


@dataclass
class CheckpointMetadata:
    """チェックポイントメタデータ"""
    checkpoint_id: str
    timestamp: str
    phase: str
    step_count: int
    progress: float
    description: str
    file_size_bytes: int
    artifacts_count: int
    hash_md5: str


class CheckpointManager:
    """チェックポイント管理システム"""
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # 5分間隔
        self.max_checkpoints = 10  # 最大保持数
        
        # メタデータ読み込み
        self.checkpoints_metadata: Dict[str, CheckpointMetadata] = {}
        self._load_metadata()
        
        self.last_auto_save = 0.0
    
    def _load_metadata(self):
        """チェックポイントメタデータを読み込み"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for checkpoint_id, metadata_dict in data.items():
                    self.checkpoints_metadata[checkpoint_id] = CheckpointMetadata(**metadata_dict)
        except Exception as e:
            print(f"⚠️ チェックポイントメタデータ読み込み失敗: {e}")
    
    def _save_metadata(self):
        """チェックポイントメタデータを保存"""
        try:
            data = {}
            for checkpoint_id, metadata in self.checkpoints_metadata.items():
                data[checkpoint_id] = asdict(metadata)
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ チェックポイントメタデータ保存失敗: {e}")
    
    def _generate_checkpoint_id(self, phase: str) -> str:
        """チェックポイントIDを生成"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        phase_short = phase.replace(" ", "_").replace("-", "_")[:20]
        return f"{timestamp}_{phase_short}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """ファイルのMD5ハッシュを計算"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def create_checkpoint(self, 
                         progress_manager: ProgressManager, 
                         description: str = "",
                         include_artifacts: bool = True,
                         force: bool = False) -> str:
        """チェックポイントを作成"""
        if not force and not self._should_create_checkpoint():
            return ""
        
        try:
            state = progress_manager.state
            checkpoint_id = self._generate_checkpoint_id(state.phase)
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            checkpoint_path.mkdir(exist_ok=True)
            
            # 進捗状態を保存
            progress_data = {
                "current_state": asdict(state),
                "phase_history": [asdict(h) for h in progress_manager.phase_history],
                "overall_progress": progress_manager.get_overall_progress()
            }
            
            progress_file = checkpoint_path / "progress.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            # アーティファクトをコピー
            artifacts_count = 0
            if include_artifacts:
                artifacts_count = self._backup_artifacts(checkpoint_path)
            
            # システム状態を保存
            system_data = {
                "python_version": sys.version,
                "working_directory": str(Path.cwd()),
                "environment_variables": {
                    k: v for k, v in os.environ.items() 
                    if k.startswith(('LLM_', 'OPENAI_', 'ANTHROPIC_', 'FAST_', 'USE_'))
                },
                "checkpoint_time": datetime.now(timezone.utc).isoformat()
            }
            
            system_file = checkpoint_path / "system.json"
            with open(system_file, 'w', encoding='utf-8') as f:
                json.dump(system_data, f, ensure_ascii=False, indent=2)
            
            # チェックポイント全体のサイズとハッシュを計算
            total_size = sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file())
            
            # メタデータ作成
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                phase=state.phase,
                step_count=state.current_step,
                progress=state.progress,
                description=description or f"自動保存: {state.phase}",
                file_size_bytes=total_size,
                artifacts_count=artifacts_count,
                hash_md5=self._calculate_file_hash(progress_file)
            )
            
            self.checkpoints_metadata[checkpoint_id] = metadata
            self._save_metadata()
            
            # 古いチェックポイントを削除
            self._cleanup_old_checkpoints()
            
            self.last_auto_save = time.time()
            
            print(f"💾 チェックポイント作成完了: {checkpoint_id}")
            print(f"   📁 サイズ: {total_size / 1024:.1f}KB, アーティファクト: {artifacts_count}個")
            
            return checkpoint_id
        
        except Exception as e:
            print(f"❌ チェックポイント作成失敗: {e}")
            return ""
    
    def _backup_artifacts(self, checkpoint_path: Path) -> int:
        """アーティファクトをバックアップ"""
        artifacts_dir = Path("data/artifacts")
        if not artifacts_dir.exists():
            return 0
        
        backup_dir = checkpoint_path / "artifacts"
        backup_dir.mkdir(exist_ok=True)
        
        count = 0
        try:
            for artifact_file in artifacts_dir.glob("*"):
                if artifact_file.is_file():
                    shutil.copy2(artifact_file, backup_dir / artifact_file.name)
                    count += 1
        except Exception as e:
            print(f"⚠️ アーティファクトバックアップで一部失敗: {e}")
        
        return count
    
    def _should_create_checkpoint(self) -> bool:
        """自動チェックポイント作成の判定"""
        if not self.auto_save_enabled:
            return False
        
        return (time.time() - self.last_auto_save) >= self.auto_save_interval
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[ProgressManager]:
        """チェックポイントから復元"""
        try:
            if checkpoint_id not in self.checkpoints_metadata:
                print(f"❌ チェックポイントが見つかりません: {checkpoint_id}")
                return None
            
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            if not checkpoint_path.exists():
                print(f"❌ チェックポイントディレクトリが存在しません: {checkpoint_path}")
                return None
            
            # 進捗状態を復元
            progress_file = checkpoint_path / "progress.json"
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            # ProgressManagerを再構築
            progress_manager = ProgressManager()
            
            # 現在の状態を復元
            current_state_dict = progress_data["current_state"]
            progress_manager.state = ProgressState(**current_state_dict)
            
            # フェーズ履歴を復元
            progress_manager.phase_history = [
                ProgressState(**hist) for hist in progress_data["phase_history"]
            ]
            
            # アーティファクトを復元
            self._restore_artifacts(checkpoint_path)
            
            print(f"✅ チェックポイント復元完了: {checkpoint_id}")
            print(f"   📊 フェーズ: {progress_manager.state.phase}")
            print(f"   📈 進捗: {int(progress_manager.state.progress * 100)}%")
            
            return progress_manager
        
        except Exception as e:
            print(f"❌ チェックポイント復元失敗: {e}")
            return None
    
    def _restore_artifacts(self, checkpoint_path: Path):
        """アーティファクトを復元"""
        backup_dir = checkpoint_path / "artifacts"
        if not backup_dir.exists():
            return
        
        artifacts_dir = Path("data/artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for backup_file in backup_dir.glob("*"):
                if backup_file.is_file():
                    shutil.copy2(backup_file, artifacts_dir / backup_file.name)
            
            print(f"   📁 アーティファクト復元完了")
        except Exception as e:
            print(f"⚠️ アーティファクト復元で一部失敗: {e}")
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """チェックポイント一覧を取得"""
        return sorted(
            self.checkpoints_metadata.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """チェックポイントを削除"""
        try:
            if checkpoint_id not in self.checkpoints_metadata:
                print(f"❌ チェックポイントが見つかりません: {checkpoint_id}")
                return False
            
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            
            del self.checkpoints_metadata[checkpoint_id]
            self._save_metadata()
            
            print(f"🗑️ チェックポイント削除完了: {checkpoint_id}")
            return True
        
        except Exception as e:
            print(f"❌ チェックポイント削除失敗: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """古いチェックポイントをクリーンアップ"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.max_checkpoints:
            # 古いものから削除
            for checkpoint in checkpoints[self.max_checkpoints:]:
                self.delete_checkpoint(checkpoint.checkpoint_id)
    
    def get_checkpoint_size(self) -> Dict[str, Any]:
        """チェックポイント全体のサイズ情報を取得"""
        total_size = 0
        checkpoint_count = len(self.checkpoints_metadata)
        
        try:
            for checkpoint_path in self.checkpoint_dir.iterdir():
                if checkpoint_path.is_dir():
                    total_size += sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file())
        except Exception:
            pass
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "checkpoint_count": checkpoint_count,
            "max_checkpoints": self.max_checkpoints,
            "auto_save_enabled": self.auto_save_enabled,
            "auto_save_interval": self.auto_save_interval
        }


def create_checkpoint_callback(checkpoint_manager: CheckpointManager) -> Callable:
    """チェックポイント作成用コールバックを作成"""
    def checkpoint_callback(state: ProgressState):
        # フェーズ完了時に自動チェックポイント作成
        if state.status == "completed":
            checkpoint_manager.create_checkpoint(
                progress_manager=None,  # コールバック内では直接アクセスできないため
                description=f"フェーズ完了: {state.phase}",
                force=True
            )
    
    return checkpoint_callback


def setup_auto_checkpoint(progress_manager: ProgressManager, 
                         interval_minutes: int = 5,
                         max_checkpoints: int = 10) -> CheckpointManager:
    """自動チェックポイント設定"""
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.auto_save_enabled = True
    checkpoint_manager.auto_save_interval = interval_minutes * 60
    checkpoint_manager.max_checkpoints = max_checkpoints
    
    # コールバック登録
    callback = create_checkpoint_callback(checkpoint_manager)
    progress_manager.add_callback(callback)
    
    return checkpoint_manager
