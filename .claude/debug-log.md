# ProjectChimera デバッグログ

## [2025-06-15] Python Package Import 問題解決

**症状**: `from project_chimera.core.container import get_container` でModuleNotFoundError
**環境**: Python 3.11, Poetry 1.8.2
**再現手順**: 
1. `poetry install` 実行後
2. `python -c "from project_chimera.core.container import get_container"`
3. ModuleNotFoundError発生

**試行錯誤**:
- ❌ `sys.path.append()` → 一時的解決だが根本的でない
- ❌ `PYTHONPATH` 環境変数設定 → Docker環境で問題
- ✅ `poetry install --no-root` → `poetry install` → 完全解決

**最終解決方法**:
```bash
# 1. キャッシュクリア
poetry cache clear pypi --all

# 2. 仮想環境再作成
poetry env remove python
poetry env use python3.11

# 3. パッケージ再インストール 
poetry install

# 4. 動作確認
poetry run python -c "from project_chimera.core.container import get_container; print('OK')"
```

**根本原因**: Poetry の editable install が不完全な状態
**予防策**: CI/CD パイプラインに import チェック追加

---

## [2025-06-14] AsyncIO WebSocket デッドロック

**症状**: WebSocket接続時に `await websocket.recv()` でハング、KeyboardInterrupt でも停止不可
**環境**: Python 3.11, websockets 12.0, Docker container
**再現手順**:
1. `BitgetWebSocketClient.connect()` 実行
2. 認証成功後、`listen()` メソッド呼び出し
3. 数分後に全スレッドがハング

**試行錯誤**:
- ❌ `asyncio.wait_for()` でタイムアウト → 根本解決せず
- ❌ `websockets.ping()` 定期実行 → メモリリーク発生
- ✅ 適切な例外ハンドリング + graceful shutdown → 解決

**最終解決方法**:
```python
async def listen(self):
    try:
        while self.is_connected:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=30.0  # 30秒タイムアウト
                )
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                # タイムアウト時はping送信
                await self.websocket.ping()
                continue
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        self.is_connected = False
        await self._cleanup()
```

**根本原因**: WebSocket の heartbeat 機能不備、例外処理不十分
**予防策**: 
- 必ず timeout 付きの `wait_for()` 使用
- Connection状態の適切な管理
- Graceful shutdown の実装

---

## [2025-06-13] TimescaleDB パフォーマンス劣化

**症状**: 価格データ挿入で 1件あたり100ms、大量データで timeout
**環境**: TimescaleDB 2.11, PostgreSQL 15, Docker
**再現手順**:
1. 1000件の価格データを連続挿入
2. 挿入時間が線形増加
3. 500件目以降で極端に遅延

**試行錯誤**:
- ❌ インデックス追加 → 改善限定的
- ❌ バッチサイズ調整 → 効果なし
- ✅ COPY + bulk insert → 100倍高速化

**最終解決方法**:
```python
# Before: 1件ずつinsert (遅い)
for price_data in price_list:
    await db.execute_query(
        "INSERT INTO price_data VALUES (%s, %s, %s)",
        (price_data.timestamp, price_data.pair, price_data.price)
    )

# After: bulk insert (高速)
async def bulk_insert_prices(self, price_data_list: List[PriceData]):
    values = [
        (data.timestamp, data.pair, data.price, data.volume)
        for data in price_data_list
    ]
    
    query = """
    INSERT INTO trading.price_data (timestamp, pair, price, volume)
    VALUES %s
    ON CONFLICT (timestamp, pair) DO UPDATE SET
        price = EXCLUDED.price,
        volume = EXCLUDED.volume
    """
    
    await self.execute_values(query, values, page_size=1000)
```

**根本原因**: PostgreSQL の単発 INSERT は overhead 大
**予防策**: 
- 100件以上は必ず bulk insert
- upsert による重複対応
- TimescaleDB圧縮ポリシー活用

---

## [2025-06-12] Docker Memory Limit 超過

**症状**: `docker compose up` 後、30分でOOM Killed
**環境**: Docker 24.0, 8GB RAM macOS
**再現手順**:
1. `docker compose up -d` 
2. 全サービス起動後、正常動作
3. 30分後にPostgreSQLコンテナがOOM Kill

**試行錯誤**:
- ❌ PostgreSQL shared_buffers 削減 → パフォーマンス劣化
- ❌ コンテナ再起動 → 根本解決せず  
- ✅ メモリ制限 + swap設定 → 安定動作

**最終解決方法**:
```yaml
# docker-compose.yml
services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    deploy:
      resources:
        limits:
          memory: 2G        # メモリ上限設定
        reservations:
          memory: 1G        # 最小保証
    environment:
      - POSTGRES_SHARED_BUFFERS=512MB     # PostgreSQL調整
      - POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
      
  app:
    build: .
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

**システム設定**:
```bash
# Docker Desktop memory 増加
# Settings > Resources > Memory: 6GB

# swappiness 調整 (Linux)
echo 'vm.swappiness=10' >> /etc/sysctl.conf
```

**根本原因**: PostgreSQL の aggressive caching + Python memory growth
**予防策**: 
- コンテナメモリ制限必須設定
- アプリケーションでのメモリリーク対策
- 定期メモリ使用量監視

---

## [2025-06-11] CI/CD Pipeline でテスト失敗

**症状**: ローカルで成功するテストがGitHub Actionsで失敗
**環境**: GitHub Actions, Ubuntu 22.04, Python 3.9-3.11 matrix
**再現手順**:
1. `git push` でCI実行
2. `test_api_client.py` で Connection timeout
3. Python 3.10 のみ失敗、3.9/3.11は成功

**試行錯誤**:
- ❌ timeout 値を増加 → 他のテストが遅延
- ❌ retry回数増加 → 根本解決せず
- ✅ モックサーバー使用 → 安定化

**最終解決方法**:
```python
# test_api_client.py
@pytest.fixture
async def mock_bitget_server():
    """モックサーバーセットアップ"""
    
    class MockServer:
        def __init__(self):
            self.app = FastAPI()
            self.setup_routes()
            
        def setup_routes(self):
            @self.app.get("/api/v2/mix/market/ticker")
            async def mock_ticker():
                return {
                    "code": "00000",
                    "data": [{
                        "symbol": "BTCUSDT_UMCBL",
                        "lastPr": "50000.0",
                        "bidPr": "49999.0", 
                        "askPr": "50001.0"
                    }]
                }
    
    server = MockServer()
    
    config = uvicorn.Config(server.app, host="127.0.0.1", port=8888)
    server_instance = uvicorn.Server(config)
    
    task = asyncio.create_task(server_instance.serve())
    await asyncio.sleep(0.1)  # サーバー起動待機
    
    yield "http://127.0.0.1:8888"
    
    server_instance.should_exit = True
    await task

# テスト実装
async def test_api_client_ticker(mock_bitget_server):
    settings = get_settings()
    settings.api.base_url = mock_bitget_server  # モックサーバー使用
    
    async with AsyncBitgetClient(settings) as client:
        ticker = await client.get_futures_ticker("BTCUSDT")
        assert ticker.price == 50000.0
```

**CI設定調整**:
```yaml
# .github/workflows/ci.yml
- name: Run tests with mock
  run: |
    poetry run pytest tests/ -v --tb=short
  env:
    BITGET_API_KEY: test_key
    BITGET_SECRET_KEY: test_secret
    BITGET_PASSPHRASE: test_passphrase
    BITGET_SANDBOX: true
    USE_MOCK_SERVER: true  # モック使用指示
```

**根本原因**: 外部API依存テストのネットワーク不安定性
**予防策**: 
- 統合テスト以外は外部依存を排除
- モックサーバーによる制御された環境
- flaky テストの自動retry設定

---

## [2025-06-10] Pydantic v2 Migration エラー

**症状**: `pydantic.v1` import エラー、設定クラス初期化失敗
**環境**: Pydantic 2.5.0, FastAPI 0.104
**再現手順**:
1. 依存関係を更新 (`poetry update`)
2. `from pydantic import BaseSettings` でImportError
3. 既存設定クラスでValidationError

**試行錯誤**:
- ❌ Pydantic v1固定 → 他ライブラリとの競合
- ❌ 部分的移行 → 中途半端で複雑化
- ✅ 完全v2移行 + 設定リファクタ → クリーンな解決

**最終解決方法**:
```python
# Before: Pydantic v1
from pydantic import BaseSettings

class Settings(BaseSettings):
    api_key: str
    
    class Config:
        env_file = ".env"

# After: Pydantic v2  
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

# ネストした設定の移行
class APIConfig(BaseModel):
    bitget_api_key: str
    bitget_secret_key: str

class Settings(BaseSettings):
    api: APIConfig
    
    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__"  # API__BITGET_API_KEY 
    }
```

**依存関係更新**:
```toml
# pyproject.toml
[tool.poetry.dependencies]
pydantic = {extras = ["dotenv"], version = "^2.5.0"}
pydantic-settings = "^2.1.0"  # 新しいパッケージ
```

**根本原因**: Pydantic v2の破壊的変更、BaseSettings分離
**予防策**: 
- メジャーバージョンアップ前の影響調査
- 段階的移行よりも一括移行
- 型チェックによる早期エラー発見

---

## [2025-06-09] Redis Connection Pool 枯渇

**症状**: Redis接続で `ConnectionError: Cannot connect to Redis`
**環境**: Redis 7.0, redis-py 5.0.1, 高負荷時発生
**再現手順**:
1. 複数プロセスで同時Redis操作
2. 10分後から間欠的にConnectionError
3. Redis server自体は正常稼働

**試行錯誤**:
- ❌ Redis memory 増加 → 効果なし
- ❌ timeout 調整 → 改善限定的
- ✅ Connection pool サイズ最適化 → 完全解決

**最終解決方法**:
```python
# Before: デフォルト設定
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# After: Connection pool 設定
import redis
from redis.connection import ConnectionPool

pool = ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=20,      # 最大接続数
    retry_on_timeout=True,   # タイムアウト時retry
    socket_keepalive=True,   # keep-alive有効
    socket_keepalive_options={},
    health_check_interval=30  # ヘルスチェック間隔
)

redis_client = redis.Redis(connection_pool=pool)

# AsyncIO対応版
import aioredis

async def create_redis_client():
    return await aioredis.from_url(
        "redis://localhost:6379",
        max_connections=20,
        retry_on_timeout=True
    )
```

**Redis設定調整**:
```conf
# redis.conf
maxclients 10000          # 最大クライアント数
timeout 300               # アイドルタイムアウト
tcp-keepalive 60          # TCP keep-alive
```

**根本原因**: デフォルトconnection pool設定が不十分
**予防策**: 
- Connection pool監視
- 接続数アラート設定
- 適切なconnection lifecycle管理

---

## [2025-06-08] JSON Serialization エラー

**症状**: `Object of type datetime is not JSON serializable`
**環境**: Python標準json、datetime objects
**再現手順**:
1. データベースからTimestamp取得
2. JSON response作成時にserializationエラー
3. APIエンドポイントで500エラー

**試行錯誤**:
- ❌ strftime()で個別変換 → 漏れが発生
- ❌ 個別にjson.dumps()引数調整 → 保守困難
- ✅ カスタムJSONEncoder作成 → 統一的解決

**最終解決方法**:
```python
import json
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID

class CustomJSONEncoder(json.JSONEncoder):
    """カスタムJSONエンコーダー"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# 使用例
data = {
    "timestamp": datetime.now(),
    "price": Decimal("50000.12"),
    "uuid": UUID("12345678-1234-5678-9012-123456789012")
}

json_string = json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False)

# FastAPI での使用
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/data")
async def get_data():
    data = {"timestamp": datetime.now()}
    return JSONResponse(
        content=data,
        headers={"Content-Type": "application/json"},
    )

# または global設定
import functools
json.dumps = functools.partial(json.dumps, cls=CustomJSONEncoder, ensure_ascii=False)
```

**根本原因**: Python標準jsonのdatetime非対応
**予防策**: 
- プロジェクト開始時にCustomEncoder設定
- API response検査の自動化
- 型安全性確保（Pydantic等）

---

## 定期メンテナンス項目

### 週次チェック
- [ ] ログファイルサイズ確認 (>1GB でアラート)
- [ ] メモリリーク検査 (プロセス再起動)
- [ ] API rate limit使用状況
- [ ] データベース容量確認

### 月次クリーンアップ  
- [ ] 古いログファイル削除 (30日以上)
- [ ] 未使用Docker image削除
- [ ] 依存関係セキュリティ監査
- [ ] バックアップデータ整合性チェック

### 緊急時対応手順
1. **システム全停止**: `docker compose down`
2. **ログ確認**: `docker compose logs --tail=100`  
3. **バックアップ復旧**: `./scripts/restore.sh [backup_file]`
4. **段階的再起動**: `docker compose up -d postgres redis` → `docker compose up -d app`