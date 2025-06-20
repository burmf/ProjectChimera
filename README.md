# 🚀 ProjectChimera - Clean Architecture Trading System

This README is provided in both English and Japanese.

---

## English

### Overview
ProjectChimera is an AI-powered high-leverage trading system designed with clean architecture principles. It integrates advanced trading components and robust risk management to deliver organized and efficient trading operations.

### Project Structure
```
ProjectChimera/
├── core/                           # Core components (e.g., API integration, risk management)
├── systems/                        # Trading systems (master, ultra bot, deployment scripts)
├── ui/                             # User interfaces (dashboards, profit maximizer, scalping tools)
├── docs/                           # Documentation
├── logs/                           # System logs
├── data/                           # Performance data
├── launch.py                       # Unified launcher
├── .env                            # Configuration file
└── README.md                       # This file
```

### Quick Start
**Option 1: Unified Launcher (Recommended)**
```bash
python3 launch.py
```

**Option 2: Direct Component Launch**
```bash
# Master Trading System
cd systems && python3 master_profit_system.py

# Dashboard
cd ui && streamlit run unified_profit_dashboard.py

# Ultra Trading Bot
cd systems && python3 ultra_trading_bot.py
```

### System Components
- **Core Components:** API client for Bitget Futures and advanced risk manager.
- **Trading Systems:** 
  - *Master System:* 40x leverage with daily targets.
  - *Ultra Bot:* 30x leverage for high-frequency trading.
  - *Deployment:* 24/7 automated management.
- **User Interfaces:** Unified Dashboard, Profit Maximizer, Scalping Dashboard.

### Performance Targets
| System | Leverage | Daily Target | Monthly ROI |
|--------|----------|--------------|-------------|
| Master | 40x      | $1,000+      | 16.8%       |
| Ultra  | 30x      | $750+        | 12.6%       |

### Safety Features
- Maximum 10% portfolio drawdown.
- Dynamic position sizing (Kelly Criterion).
- Real-time risk monitoring and automatic emergency stops.

### Benefits of Clean Architecture
- Modular design for easy maintenance and scalability.
- Clear separation between core, systems, and UI layers.
- Organized logging and easy deployment.

### Start Trading
```bash
python3 launch.py
```
Begin trading and start generating profits!

---

## 日本語

### 概要
ProjectChimera は、クリーンアーキテクチャの原則に基づいて設計された、AI搭載の高レバレッジトレーディングシステムです。先進の取引コンポーネントと堅牢なリスク管理を統合し、効率的かつ組織的なトレーディングを実現します。

### プロジェクト構成
```
ProjectChimera/
├── core/                           # コアコンポーネント（例：API連携、リスク管理）
├── systems/                        # トレーディングシステム（マスターシステム、ウルトラボット、デプロイ用スクリプト）
├── ui/                             # ユーザーインターフェース（ダッシュボード、プロフィットマキシマイザー、スキャルピングツール）
├── docs/                           # ドキュメント
├── logs/                           # ログ
├── data/                           # パフォーマンスデータ
├── launch.py                       # 統合ランチャー
├── .env                            # 設定ファイル
└── README.md                       # このファイル
```

### クイックスタート
【オプション 1: 統合ランチャー（推奨）】
```bash
python3 launch.py
```

【オプション 2: 個別コンポーネントの起動】
```bash
# マスターシステム
cd systems && python3 master_profit_system.py

# ダッシュボード
cd ui && streamlit run unified_profit_dashboard.py

# ウルトラトレーディングボット
cd systems && python3 ultra_trading_bot.py
```

### システム構成
- **コアコンポーネント:** Bitget Futures 用 API クライアントおよび高度なリスク管理システム。
- **トレーディングシステム:** 
  - *マスターシステム:* 40倍のレバレッジ、日々の目標達成。
  - *ウルトラボット:* 30倍のレバレッジによる高頻度取引。
  - *デプロイメント:* 24時間365日の自動管理。
- **ユーザーインターフェース:** 統合ダッシュボード、プロフィットマキシマイザー、スキャルピングダッシュボード。

### パフォーマンス目標
| システム  | レバレッジ | 日々の目標   | 月間 ROI   |
|-----------|----------|--------------|------------|
| マスター  | 40x      | $1,000+      | 16.8%      |
| ウルトラ  | 30x      | $750+        | 12.6%      |

### 安全対策
- 最大 10% のポートフォリオドローダウン。
- ケリー基準による動的なポジションサイジング。
- リアルタイムのリスク監視と自動緊急停止機能。

### クリーンアーキテクチャの利点
- 維持管理と拡張が容易なモジュラーデザイン。
- コア、システム、UI を明確に分離。
- 整理されたログ管理と簡単なデプロイメント。

### トレーディング開始方法
```bash
python3 launch.py
```
トレーディングを開始し、利益を追求しましょう！

---

Clean code, maximum profits! / クリーンなコードで、最大限の利益を！
