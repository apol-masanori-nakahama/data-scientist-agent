# 統合進捗システム

Data Scientist Agent に包括的な進捗表示・管理システムを導入しました。どの瞬間でも進捗状況を確認でき、処理の透明性を大幅に向上させています。

## 🚀 機能概要

### 1. リアルタイム進捗表示
- **プログレスバー**: 現在のフェーズとステップの進捗率を視覚的に表示
- **ETA計算**: 過去のパフォーマンスから残り時間を予測
- **ステップ詳細**: 現在実行中のタスクの詳細情報
- **フェーズ履歴**: 完了したフェーズの一覧と所要時間

### 2. 包括的ログシステム
- **構造化ログ**: JSON形式でタイムスタンプ、進捗情報、エラー詳細を記録
- **複数出力**: コンソール、ファイル、Streamlitダッシュボードに同時出力
- **ログレベル管理**: 詳細度を調整可能

### 3. 通知システム
- **メール通知**: 重要なマイルストーンやエラー時にメール送信
- **デスクトップ通知**: OS標準の通知機能を使用
- **Webhook連携**: SlackやDiscordなどへの通知
- **サウンド通知**: 音声による進捗アラート

### 4. チェックポイント機能
- **自動保存**: 定期的な進捗状態の保存
- **手動チェックポイント**: 重要なポイントでの手動保存
- **復元機能**: 過去の状態から処理を再開
- **アーティファクト管理**: 生成されたファイルも含めて保存

### 5. Streamlitダッシュボード
- **リアルタイム更新**: 進捗状況の自動更新表示
- **インタラクティブ**: チェックポイント作成・復元の操作
- **視覚的表示**: グラフやメトリクスでの進捗可視化

## 📋 使用方法

### Streamlit UI モード

```bash
streamlit run app.py
```

新しく追加された「進捗」タブで以下が確認できます：
- リアルタイム進捗ダッシュボード
- チェックポイント管理
- 詳細進捗情報
- システム設定
- 進捗ログ

### CLI モード

```bash
CLI=1 python app.py data/sample.csv
```

コンソールに詳細な進捗情報が表示されます：
- フェーズ別進捗バー
- ETA情報
- ステップ詳細
- 完了サマリー

## ⚙️ 設定

### 環境変数

#### 通知設定
```bash
# メール通知
export NOTIFICATION_EMAIL_ENABLED=true
export NOTIFICATION_EMAIL_FROM=your-email@gmail.com
export NOTIFICATION_EMAIL_TO=recipient@gmail.com
export NOTIFICATION_EMAIL_PASSWORD=your-app-password
export NOTIFICATION_EMAIL_SMTP=smtp.gmail.com
export NOTIFICATION_EMAIL_PORT=587

# Webhook通知（Slack等）
export NOTIFICATION_WEBHOOK_ENABLED=true
export NOTIFICATION_WEBHOOK_URL=https://hooks.slack.com/services/...

# デスクトップ通知
export NOTIFICATION_DESKTOP_ENABLED=true

# サウンド通知
export NOTIFICATION_SOUND_ENABLED=true
```

#### 進捗システム設定
```bash
# ダッシュボード有効化
export PROGRESS_DASHBOARD=true

# 通知有効化
export PROGRESS_NOTIFICATIONS=true

# チェックポイント有効化
export PROGRESS_CHECKPOINTS=true

# コンソール出力有効化
export PROGRESS_CONSOLE=true

# チェックポイント保存間隔（分）
export CHECKPOINT_INTERVAL_MINUTES=5
```

## 📁 ファイル構成

### 新規追加ファイル

```
src/utils/
├── progress.py          # 進捗管理コアシステム
├── dashboard.py         # Streamlitダッシュボード
├── notifications.py     # 通知システム
├── checkpoint.py        # チェックポイント機能
├── integrated_progress.py # 統合システム
└── logging.py          # 強化されたログシステム
```

### 生成されるファイル

```
data/
├── artifacts/
│   ├── progress.json     # 進捗状態保存
│   ├── progress.log      # 進捗ログ
│   ├── dashboard.json    # ダッシュボードスナップショット
│   └── notifications.log # 通知ログ
└── checkpoints/
    ├── checkpoints.json  # チェックポイントメタデータ
    └── [checkpoint_id]/  # 個別チェックポイント
        ├── progress.json
        ├── system.json
        └── artifacts/
```

## 🔧 API使用例

### プログラム内での使用

```python
from src.utils.integrated_progress import get_global_progress_system

# 進捗システム取得
progress = get_global_progress_system()

# フェーズ開始
progress.start_phase("データ前処理", total_steps=5, message="前処理を開始します")

# ステップ更新
progress.update_step("データ読み込み", message="CSVファイルを読み込み中")
progress.update_step("欠損値処理", message="欠損値を処理中")

# フェーズ完了
progress.complete_phase("前処理完了")

# チェックポイント作成
checkpoint_id = progress.save_checkpoint("前処理完了時点")

# 進捗状況取得
status = progress.get_overall_status()
print(f"現在の進捗: {status['progress']['current_state']['progress'] * 100:.1f}%")
```

### 独立したコンポーネント使用

```python
from src.utils.progress import ProgressManager
from src.utils.notifications import NotificationManager, NotificationType

# 進捗管理
progress_manager = ProgressManager()
progress_manager.start_phase("処理中", 10)

# 通知管理
notification_manager = NotificationManager()
notification_manager.send_notification(
    NotificationType.PHASE_START,
    "処理開始",
    "データ分析を開始しました",
    progress_manager.state
)
```

## 🎯 主な改善点

### Before（改善前）
- 基本的なStreamlitプログレスバーのみ
- エラー時の詳細情報が不足
- 処理中断時の復旧が困難
- CLI モードでの進捗表示が最小限

### After（改善後）
- **包括的進捗管理**: フェーズ、ステップ、ETA、履歴
- **多様な通知方法**: メール、デスクトップ、Webhook、音声
- **強固な復旧機能**: チェックポイントによる状態保存・復元
- **詳細なログ**: 構造化された実行ログ
- **視覚的ダッシュボード**: リアルタイム状況表示
- **CLI強化**: コンソールでも詳細な進捗表示

## 🔍 トラブルシューティング

### よくある問題

1. **通知が送信されない**
   - 環境変数が正しく設定されているか確認
   - メール認証情報（アプリパスワード）を確認

2. **チェックポイントが作成されない**
   - `data/checkpoints/` ディレクトリの書き込み権限を確認
   - ディスク容量を確認

3. **ダッシュボードが更新されない**
   - Streamlitのキャッシュをクリア: `streamlit cache clear`
   - ブラウザのリロード

4. **CLI モードで進捗が表示されない**
   - `PROGRESS_CONSOLE=true` が設定されているか確認
   - ログレベルを確認

### ログの確認

```bash
# 進捗ログを確認
tail -f data/artifacts/progress.log

# 通知ログを確認
tail -f data/artifacts/notifications.log

# システム全体のログを確認
tail -f *.log
```

## 🚀 今後の拡張予定

- **Web API**: REST APIでの進捗状況取得
- **モバイル通知**: プッシュ通知対応
- **分散処理**: 複数プロセス間での進捗同期
- **カスタムダッシュボード**: ユーザー定義の表示項目
- **パフォーマンス分析**: 処理時間の詳細分析と最適化提案

---

この統合進捗システムにより、Data Scientist Agent の使用体験が大幅に向上し、どの瞬間でも処理状況を把握できるようになりました。
