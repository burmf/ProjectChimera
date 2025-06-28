"""
ProjectChimera システム監視ダッシュボード
全システムコンポーネントのリアルタイム稼働状況を表示
"""

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import plotly.express as px
import psutil
import redis
import streamlit as st
from sqlalchemy import create_engine, text

try:
    from ..settings import get_settings
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from settings import get_settings


class SystemMonitor:
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.db_engine = None
        self.last_update = None

    def init_connections(self):
        """外部接続を初期化"""
        try:
            # Redis接続
            redis_url = self.settings.layer_system.redis_streams.redis_url
            self.redis_client = redis.from_url(redis_url, decode_responses=True)

            # DB接続
            db_url = str(self.settings.database_url)
            if db_url.startswith("sqlite"):
                # SQLiteの場合はファイルパス調整
                db_url = db_url.replace("sqlite:///", "sqlite:///data/")
            self.db_engine = create_engine(db_url)

        except Exception as e:
            st.error(f"Connection initialization failed: {e}")

    def check_system_health(self) -> dict[str, Any]:
        """システムヘルスチェック"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "resources": {},
            "alerts": [],
        }

        try:
            # システムリソース
            health["resources"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "network_io": dict(psutil.net_io_counters()._asdict()),
            }

            # コンポーネント別チェック
            health["components"]["redis"] = self._check_redis()
            health["components"]["database"] = self._check_database()
            health["components"]["openai"] = self._check_openai()
            health["components"]["bitget"] = self._check_bitget()
            health["components"]["processes"] = self._check_processes()

            # 全体ステータス判定
            failed_components = [
                name
                for name, status in health["components"].items()
                if status.get("status") != "healthy"
            ]

            if failed_components:
                health["overall_status"] = (
                    "degraded" if len(failed_components) < 3 else "critical"
                )
                health["alerts"].extend(
                    [f"{comp} is unhealthy" for comp in failed_components]
                )

            # リソースアラート
            if health["resources"]["cpu_percent"] > 80:
                health["alerts"].append("High CPU usage detected")
            if health["resources"]["memory_percent"] > 80:
                health["alerts"].append("High memory usage detected")
            if health["resources"]["disk_percent"] > 90:
                health["alerts"].append("Low disk space")

        except Exception as e:
            health["overall_status"] = "critical"
            health["alerts"].append(f"Health check error: {e}")

        return health

    def _check_redis(self) -> dict[str, Any]:
        """Redis接続チェック"""
        try:
            if not self.redis_client:
                return {"status": "disconnected", "error": "No connection"}

            # 接続テスト
            info = self.redis_client.info()

            # ストリーム情報取得
            streams = {}
            stream_names = [
                self.settings.layer_system.redis_streams.market_data_stream,
                self.settings.layer_system.redis_streams.news_stream,
                self.settings.layer_system.redis_streams.x_posts_stream,
                self.settings.layer_system.redis_streams.ai_decisions_stream,
            ]

            for stream in stream_names:
                try:
                    length = self.redis_client.xlen(stream)
                    streams[stream] = length
                except:
                    streams[stream] = 0

            return {
                "status": "healthy",
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                "streams": streams,
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_database(self) -> dict[str, Any]:
        """データベース接続チェック"""
        try:
            if not self.db_engine:
                return {"status": "disconnected", "error": "No connection"}

            with self.db_engine.connect() as conn:
                # 接続テスト
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

                # テーブル存在チェック
                tables = {}
                table_names = [
                    "market_data",
                    "news_articles",
                    "ai_decisions",
                    "executions",
                ]

                for table in table_names:
                    try:
                        count_result = conn.execute(
                            text(f"SELECT COUNT(*) FROM {table}")
                        )
                        tables[table] = count_result.scalar()
                    except Exception as e:
                        logger.warning(f"Failed to get table count for {table}: {e}")
                        tables[table] = "N/A"

                return {"status": "healthy", "connection": "active", "tables": tables}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_openai(self) -> dict[str, Any]:
        """OpenAI API設定チェック"""
        try:
            api_key = self.settings.layer_system.ai.openai_api_key.get_secret_value()
            model = self.settings.layer_system.ai.openai_model
            enabled = self.settings.layer_system.ai.enable_1min_decisions

            status = "healthy" if api_key and enabled else "disabled"

            return {
                "status": status,
                "model": model,
                "enabled": enabled,
                "has_api_key": bool(api_key),
                "daily_cost_limit": self.settings.layer_system.ai.max_daily_api_cost_usd,
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_bitget(self) -> dict[str, Any]:
        """Bitget API設定チェック"""
        try:
            api_key = self.settings.api.bitget_key.get_secret_value()
            sandbox = self.settings.api.bitget_sandbox

            status = "configured" if api_key else "not_configured"

            return {
                "status": status,
                "sandbox_mode": sandbox,
                "has_api_key": bool(api_key),
                "base_url": self.settings.api.bitget_rest_url,
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_processes(self) -> dict[str, Any]:
        """実行中プロセスチェック"""
        try:
            processes = {}

            # Python関連プロセス検索
            for proc in psutil.process_iter(
                ["pid", "name", "cmdline", "create_time", "cpu_percent"]
            ):
                try:
                    if "python" in proc.info["name"].lower():
                        cmdline = (
                            " ".join(proc.info["cmdline"])
                            if proc.info["cmdline"]
                            else ""
                        )

                        # ProjectChimera関連プロセス特定
                        if any(
                            keyword in cmdline.lower()
                            for keyword in ["streamlit", "orchestrator", "chimera"]
                        ):
                            process_name = "Unknown"
                            if "streamlit" in cmdline:
                                process_name = "Streamlit UI"
                            elif "orchestrator" in cmdline:
                                process_name = "Trading Orchestrator"

                            processes[process_name] = {
                                "pid": proc.info["pid"],
                                "status": "running",
                                "uptime_seconds": time.time()
                                - proc.info["create_time"],
                                "cpu_percent": proc.info["cpu_percent"] or 0,
                            }

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                "status": "healthy" if processes else "no_processes",
                "active_processes": processes,
                "total_count": len(processes),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_historical_metrics(self, hours: int = 24) -> dict[str, Any]:
        """過去のメトリクス取得"""
        try:
            # Redis stream データの取得
            metrics = {
                "timestamps": [],
                "message_counts": {
                    "news": [],
                    "ai_decisions": [],
                    "market_data": [],
                    "x_posts": [],
                },
            }

            # 過去24時間のタイムスタンプ生成
            now = datetime.now()
            for i in range(hours):
                ts = now - timedelta(hours=i)
                metrics["timestamps"].append(ts.isoformat())

                # 各ストリームのメッセージ数（簡易版）
                for stream_type in metrics["message_counts"]:
                    # 実際の実装では Redis stream から時間範囲でデータ取得
                    # ここでは仮の値を設定
                    metrics["message_counts"][stream_type].append(
                        max(0, 10 + (i % 5) * 2)  # サンプルデータ
                    )

            # 逆順にして時系列順に
            for key in metrics["message_counts"]:
                metrics["message_counts"][key].reverse()
            metrics["timestamps"].reverse()

            return metrics

        except Exception as e:
            return {"error": str(e)}


def render_system_monitor():
    """システム監視ダッシュボードのレンダリング"""

    st.title("🔍 ProjectChimera システム監視")
    st.markdown("全システムコンポーネントのリアルタイム稼働状況")

    # 監視オブジェクト初期化
    if "monitor" not in st.session_state:
        st.session_state.monitor = SystemMonitor()
        st.session_state.monitor.init_connections()

    monitor = st.session_state.monitor

    # 自動更新設定
    auto_refresh = st.sidebar.checkbox("自動更新 (30秒)", value=True)
    if auto_refresh:
        time.sleep(1)  # 初回表示の遅延
        st.rerun()

    # 手動更新ボタン
    if st.sidebar.button("🔄 手動更新"):
        st.rerun()

    # システムヘルス取得
    health = monitor.check_system_health()

    # 全体ステータス表示
    st.subheader("🚦 システム全体ステータス")

    status_color = {"healthy": "🟢", "degraded": "🟡", "critical": "🔴"}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "全体ステータス",
            f"{status_color.get(health['overall_status'], '⚪')} {health['overall_status'].upper()}",
            delta=None,
        )

    with col2:
        active_components = sum(
            1
            for comp in health["components"].values()
            if comp.get("status") == "healthy"
        )
        total_components = len(health["components"])
        st.metric(
            "稼働コンポーネント",
            f"{active_components}/{total_components}",
            delta=(
                f"{(active_components/total_components)*100:.0f}%"
                if total_components > 0
                else "0%"
            ),
        )

    with col3:
        alert_count = len(health["alerts"])
        st.metric(
            "アラート数", alert_count, delta="注意が必要" if alert_count > 0 else "正常"
        )

    # アラート表示
    if health["alerts"]:
        st.subheader("⚠️ アラート")
        for alert in health["alerts"]:
            st.warning(alert)

    # システムリソース
    st.subheader("💾 システムリソース")

    col1, col2, col3 = st.columns(3)

    with col1:
        cpu_percent = health["resources"]["cpu_percent"]
        st.metric(
            "CPU使用率",
            f"{cpu_percent:.1f}%",
            delta="高負荷" if cpu_percent > 80 else None,
        )

    with col2:
        mem_percent = health["resources"]["memory_percent"]
        st.metric(
            "メモリ使用率",
            f"{mem_percent:.1f}%",
            delta="高使用率" if mem_percent > 80 else None,
        )

    with col3:
        disk_percent = health["resources"]["disk_percent"]
        st.metric(
            "ディスク使用率",
            f"{disk_percent:.1f}%",
            delta="容量不足" if disk_percent > 90 else None,
        )

    # コンポーネント詳細
    st.subheader("🔧 コンポーネント詳細")

    # Redis
    redis_status = health["components"]["redis"]
    with st.expander(
        f"Redis {'🟢' if redis_status.get('status') == 'healthy' else '🔴'}"
    ):
        if redis_status.get("status") == "healthy":
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "稼働時間", f"{redis_status.get('uptime_seconds', 0)//3600:.0f}時間"
                )
                st.metric(
                    "接続クライアント数", redis_status.get("connected_clients", 0)
                )
            with col2:
                st.metric(
                    "メモリ使用量", f"{redis_status.get('used_memory_mb', 0):.1f}MB"
                )

                # ストリーム情報
                streams = redis_status.get("streams", {})
                if streams:
                    st.write("**ストリーム:**")
                    for stream, count in streams.items():
                        st.write(f"- {stream}: {count} messages")
        else:
            st.error(f"Error: {redis_status.get('error', 'Unknown error')}")

    # データベース
    db_status = health["components"]["database"]
    with st.expander(
        f"Database {'🟢' if db_status.get('status') == 'healthy' else '🔴'}"
    ):
        if db_status.get("status") == "healthy":
            tables = db_status.get("tables", {})
            if tables:
                st.write("**テーブル レコード数:**")
                for table, count in tables.items():
                    st.write(f"- {table}: {count}")
        else:
            st.error(f"Error: {db_status.get('error', 'Unknown error')}")

    # OpenAI
    openai_status = health["components"]["openai"]
    with st.expander(
        f"OpenAI {'🟢' if openai_status.get('status') == 'healthy' else '🟡' if openai_status.get('status') == 'disabled' else '🔴'}"
    ):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ステータス:** {openai_status.get('status', 'unknown')}")
            st.write(f"**モデル:** {openai_status.get('model', 'N/A')}")
        with col2:
            st.write(f"**有効:** {'✅' if openai_status.get('enabled') else '❌'}")
            st.write(
                f"**APIキー:** {'✅' if openai_status.get('has_api_key') else '❌'}"
            )
        st.write(f"**日次コスト上限:** ${openai_status.get('daily_cost_limit', 0)}")

    # Bitget
    bitget_status = health["components"]["bitget"]
    with st.expander(
        f"Bitget {'🟢' if bitget_status.get('status') == 'configured' else '🟡'}"
    ):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ステータス:** {bitget_status.get('status', 'unknown')}")
            st.write(
                f"**APIキー:** {'✅' if bitget_status.get('has_api_key') else '❌'}"
            )
        with col2:
            st.write(
                f"**サンドボックス:** {'✅' if bitget_status.get('sandbox_mode') else '❌'}"
            )
            st.write(f"**Base URL:** {bitget_status.get('base_url', 'N/A')}")

    # プロセス
    process_status = health["components"]["processes"]
    with st.expander(
        f"Running Processes {'🟢' if process_status.get('status') == 'healthy' else '🟡'}"
    ):
        processes = process_status.get("active_processes", {})
        if processes:
            for name, proc in processes.items():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{name}**")
                with col2:
                    st.write(f"PID: {proc['pid']}")
                with col3:
                    uptime_hours = proc["uptime_seconds"] / 3600
                    st.write(f"稼働: {uptime_hours:.1f}h")
        else:
            st.warning("ProjectChimera関連プロセスが見つかりません")

    # 履歴チャート
    st.subheader("📊 メトリクス履歴")

    try:
        metrics = monitor.get_historical_metrics(24)

        if "error" not in metrics:
            df = pd.DataFrame(metrics["message_counts"])
            df["timestamp"] = pd.to_datetime(metrics["timestamps"])

            fig = px.line(
                df,
                x="timestamp",
                y=["news", "ai_decisions", "market_data", "x_posts"],
                title="過去24時間のメッセージ処理数",
                labels={"value": "メッセージ数", "timestamp": "時刻"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"メトリクス取得エラー: {metrics['error']}")

    except Exception as e:
        st.error(f"チャート表示エラー: {e}")

    # 最終更新時刻
    st.caption(f"最終更新: {health['timestamp']}")


if __name__ == "__main__":
    render_system_monitor()
