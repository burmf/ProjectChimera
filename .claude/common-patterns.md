# ProjectChimera 定型パターン集

## 開発コマンドパターン

### プロジェクト初期化
```bash
# 開発環境セットアップ
poetry install --with dev
poetry shell

# Docker環境起動  
docker compose up -d
docker compose logs -f app

# 開発用設定での起動
export ENVIRONMENT=development
poetry run python -m project_chimera.systems.master_trading_system
```

### テスト実行パターン
```bash
# 全テスト実行
poetry run pytest tests/ -v --cov=project_chimera

# 統合テストのみ
poetry run pytest tests/test_integration.py -v

# 特定モジュールテスト
poetry run pytest tests/test_ai_department_simple.py -v -s

# カバレッジレポート生成
poetry run pytest --cov-report=html
open htmlcov/index.html
```

### AI部門システム操作
```bash
# AI部門協調テスト
poetry run python -c "
from core.ai_orchestrator import AIOrchestrator
from core.ai_orchestrator import MarketSituation, DecisionType
import asyncio

async def test():
    orchestrator = AIOrchestrator()
    market_data = MarketSituation(...)
    decision = await orchestrator.analyze_market_situation(market_data, DecisionType.TRADE_SIGNAL)
    print(f'Decision: {decision.final_decision}')
    
asyncio.run(test())
"
```

### バックテスト実行
```bash
# 基本バックテスト
poetry run chimera backtest --csv data/btcusdt_1m.csv --strategy ma_crossover

# Kelly基準戦略
poetry run chimera backtest --csv data/ethusdt_1m.csv --strategy kelly --initial-capital 50000

# 結果保存
poetry run chimera backtest --csv data/btcusdt_1m.csv --strategy ma_crossover --output results.json
```

### データベース操作
```bash
# PostgreSQL接続
docker exec -it projectchimera-postgres-1 psql -U trading_user -d trading_db

# データベースマイグレーション
poetry run python scripts/migrate_sqlite_to_postgres.py

# バックアップ作成
./scripts/backup.sh
```

## よく使用する開発パターン

### 新しいAI部門作成
```python
# 1. 基底クラス継承
from core.ai_agent_base import AIAgentBase
from departments.department_coordination import DepartmentType

class NewAnalysisAI(AIAgentBase):
    def __init__(self):
        super().__init__()
        self.department = DepartmentType.NEW_ANALYSIS
    
    async def analyze(self, market_data: MarketSituation) -> DepartmentDecision:
        # 専門分析実装
        analysis_result = await self._perform_analysis(market_data)
        
        return DepartmentDecision(
            department=self.department,
            decision=analysis_result,
            confidence=0.85,
            reasoning="専門分析に基づく判断"
        )

# 2. オーケストレーターに登録
orchestrator.register_department(DepartmentType.NEW_ANALYSIS, NewAnalysisAI())
```

### 新しいリスク指標追加
```python
# risk_manager.py に追加
def calculate_new_risk_metric(self) -> float:
    """新しいリスク指標の計算"""
    # 実装
    
    # メトリクス更新
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        # 既存メトリクスに追加
        new_metric = self.calculate_new_risk_metric()
        
        return PortfolioMetrics(
            # 既存フィールド
            new_risk_metric=new_metric  # 新指標追加
        )
```

### API クライアント拡張
```python
# api_client.py に新しいエンドポイント追加
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionException, RateLimitException))
)
async def new_api_method(self, params: Dict) -> Dict:
    """新しいAPI呼び出し"""
    endpoint = '/api/v2/new/endpoint'
    
    try:
        result = await self._make_request('GET', endpoint, params=params)
        return self._process_new_response(result)
        
    except Exception as e:
        logger.error(f"New API call failed: {e}")
        raise
```

### 構造化ログ追加
```python
# 新しいイベントタイプ定義
class EventType(Enum):
    NEW_EVENT = "new_event"

# ログ出力
structured_logger.log_event(
    EventType.NEW_EVENT,
    "新しいイベントが発生",
    extra={
        "custom_field": "カスタム値",
        "metric_value": 123.45
    }
)
```

## トラブルシューティングパターン

### Docker 関連問題
```bash
# コンテナ状態確認
docker compose ps
docker compose logs app

# 完全リセット
docker compose down -v
docker system prune -a
docker compose up -d --build

# データベース接続確認
docker exec -it projectchimera-postgres-1 pg_isready
```

### API接続問題
```bash
# API キー確認
echo $BITGET_API_KEY | head -c 10

# 接続テスト
poetry run python -c "
from project_chimera.core.api_client import AsyncBitgetClient
from project_chimera.config import get_settings
import asyncio

async def test():
    async with AsyncBitgetClient(get_settings()) as client:
        ticker = await client.get_futures_ticker('BTCUSDT')
        print(f'BTC Price: {ticker.price if ticker else \"Failed\"}')

asyncio.run(test())
"
```

### メモリリーク調査
```bash
# メモリ使用量監視
poetry run python -c "
import psutil
import time
from project_chimera.systems.master_trading_system import main

process = psutil.Process()
while True:
    print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
    time.sleep(60)
"
```

### パフォーマンス測定
```python
# レイテンシー測定
import time
from loguru import logger

async def measure_latency(func, *args, **kwargs):
    start = time.time()
    result = await func(*args, **kwargs)
    latency = (time.time() - start) * 1000
    logger.info(f"Function {func.__name__} latency: {latency:.2f}ms")
    return result

# 使用例
ticker = await measure_latency(client.get_futures_ticker, 'BTCUSDT')
```

## 設定管理パターン

### 環境別設定切り替え
```bash
# 開発環境
export ENVIRONMENT=development
export DEBUG=true

# 本番環境  
export ENVIRONMENT=production
export DEBUG=false
export BITGET_SANDBOX=false
```

### 設定値のオーバーライド
```python
# 一時的な設定変更
from project_chimera.config import get_settings

settings = get_settings()
settings.trading.base_leverage = 5  # デフォルトから変更
```

### YAML設定ファイル使用
```bash
# 設定ファイル生成
poetry run python -c "
from project_chimera.config import create_config_template
create_config_template('development')
create_config_template('production')
"

# YAML設定での起動
poetry run python -c "
from project_chimera.config import load_config_from_yaml
settings = load_config_from_yaml('config/production.yaml')
"
```

## デバッグパターン

### ステップバイステップデバッグ
```python
# 1. ログレベル調整
import logging
logging.getLogger().setLevel(logging.DEBUG)

# 2. ブレークポイント設定
import pdb; pdb.set_trace()

# 3. 変数ダンプ
from pprint import pprint
pprint(vars(object))
```

### AI判断過程の可視化
```python
# 部門別判断の詳細確認
decision = await orchestrator.analyze_market_situation(market_data, DecisionType.TRADE_SIGNAL)

for dept_decision in decision.department_decisions:
    print(f"{dept_decision.department}: {dept_decision.confidence:.3f}")
    print(f"  Reasoning: {dept_decision.reasoning}")
```

### リスク計算の検証
```python
# VaR計算の手法別比較
var_mc = risk_manager.calculate_portfolio_var(method="monte_carlo")
var_param = risk_manager.calculate_portfolio_var(method="parametric")
var_hist = risk_manager.calculate_portfolio_var(method="historical")

print(f"Monte Carlo VaR: ${var_mc['var_95']:,.2f}")
print(f"Parametric VaR: ${var_param['var_95']:,.2f}")  
print(f"Historical VaR: ${var_hist['var_95']:,.2f}")
```

## 運用監視パターン

### ヘルスチェック
```python
# システム状態確認
async def health_check():
    checks = {
        "database": await db_health_check(),
        "api": await api_health_check(), 
        "redis": await redis_health_check()
    }
    
    all_healthy = all(checks.values())
    return {"status": "healthy" if all_healthy else "unhealthy", "checks": checks}
```

### メトリクス収集
```python
# パフォーマンスメトリクス
structured_logger.log_performance_metric(
    "api_latency", latency_ms, "milliseconds", 
    component="bitget_client"
)

structured_logger.log_performance_metric(
    "memory_usage", memory_mb, "megabytes",
    component="trading_engine"
)
```

### アラート設定
```python
# 異常検知とアラート
if portfolio_metrics.current_drawdown < -0.10:  # 10%以上のドローダウン
    structured_logger.log_security_event(
        "Large drawdown detected", 
        severity="high",
        extra={"drawdown": portfolio_metrics.current_drawdown}
    )
    
    # 緊急停止処理
    await emergency_stop_trading()
```

## 本番デプロイパターン

### Zero-downtime デプロイ
```bash
# 1. 新バージョンビルド
docker compose build app

# 2. ローリング更新
docker compose up -d --no-deps app

# 3. ヘルスチェック待機
curl -f http://localhost:8501/health || exit 1

# 4. 旧バージョンクリーンアップ
docker image prune -f
```

### 設定ファイル更新
```bash
# 設定変更をGitで管理
git add config/production.yaml
git commit -m "Update production config"

# 本番環境に反映
kubectl create configmap app-config --from-file=config/production.yaml
kubectl rollout restart deployment/trading-app
```

### データベースマイグレーション
```bash
# バックアップ作成
./scripts/backup.sh

# マイグレーション実行
poetry run python scripts/migrate_schema.py

# 整合性チェック
poetry run python scripts/verify_migration.py
```

## チーム開発パターン

### コードレビュー前チェック
```bash
# 品質チェック一括実行
poetry run ruff check project_chimera/
poetry run black --check project_chimera/
poetry run mypy project_chimera/
poetry run pytest tests/ --cov=project_chimera
```

### 機能ブランチ開発
```bash
# 機能ブランチ作成
git checkout develop
git pull origin develop
git checkout -b feature/new-ai-department

# 開発完了後
git add .
git commit -m "feat(ai): Add sentiment analysis department"
git push origin feature/new-ai-department

# プルリクエスト作成
gh pr create --title "Add sentiment analysis AI department" --body "Implements news sentiment analysis with 85% accuracy"
```

### 知見更新コマンド
```bash
# 重要な実装完了時
echo "新しい実装パターン・設計決定を .claude/ ファイルに記録してください"

# 問題解決時  
echo "トラブルシューティングの手順を .claude/debug-log.md に追加してください"

# 改善完了時
echo "改善内容と効果を .claude/project-improvements.md に記録してください"
```