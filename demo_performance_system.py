#!/usr/bin/env python3
"""
パフォーマンス測定システムデモ
取引ロジックの完成度PDCAサイクル実演
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
import json

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.project_chimera.monitor.performance_logger import (
    initialize_performance_logging, TradeExecution, TradeResult, PerformancePhase
)
from src.project_chimera.monitor.pdca_analyzer import get_pdca_analyzer
from src.project_chimera.strategies.enhanced_weekend_effect import create_enhanced_weekend_effect_strategy
from src.project_chimera.domains.market import MarketFrame


class PerformanceSystemDemo:
    """パフォーマンス測定システムデモクラス"""
    
    def __init__(self):
        self.performance_logger = None
        self.pdca_analyzer = None
        self.strategy = None
        
    async def initialize(self):
        """システム初期化"""
        print("🚀 Initializing Performance Measurement System...")
        
        # パフォーマンスロガー初期化
        self.performance_logger = await initialize_performance_logging("logs/demo_performance")
        self.pdca_analyzer = get_pdca_analyzer(self.performance_logger)
        
        # Enhanced Weekend Effect Strategy作成
        self.strategy = create_enhanced_weekend_effect_strategy({
            "confidence_threshold": 0.6,
            "base_position_size": 1.0
        })
        
        print("✅ Performance measurement system initialized")
        print(f"📊 Session ID: {self.performance_logger.current_session_id}")
        
    async def simulate_trading_session(self, num_trades: int = 50):
        """取引セッションシミュレーション"""
        print(f"\n📈 Simulating {num_trades} trades...")
        
        base_time = datetime.now()
        base_price = 50000.0
        
        for i in range(num_trades):
            # マーケットデータ生成
            timestamp = base_time + timedelta(hours=i*2)
            price_change = random.uniform(-0.02, 0.02)  # ±2%の価格変動
            current_price = base_price * (1 + price_change)
            
            # MarketFrame作成（簡易版）
            market_frame = MarketFrame(
                symbol="BTCUSDT",
                timestamp=timestamp.timestamp(),
                bid=current_price - 10,
                ask=current_price + 10,
                last=current_price,
                volume=random.uniform(1000, 10000),
                source="demo"
            )
            
            # 取引データ生成
            trade_execution = TradeExecution(
                trade_id=f"demo_trade_{i+1:03d}",
                strategy_name="EnhancedWeekendEffectStrategy",
                timestamp=timestamp,
                symbol="BTCUSDT",
                signal_type=random.choice(["buy", "sell"]),
                signal_confidence=random.uniform(0.4, 0.9),
                signal_generation_time_ms=random.uniform(50, 500),
                entry_price=current_price,
                exit_price=current_price * (1 + random.uniform(-0.01, 0.015)),
                quantity=random.uniform(0.5, 2.0),
                commission=random.uniform(1, 10),
                pnl=random.uniform(-50, 100),
                pnl_percentage=random.uniform(-2, 4),
                hold_duration_seconds=random.uniform(1800, 7200),  # 30分-2時間
                order_placement_time_ms=random.uniform(100, 1000),
                fill_time_ms=random.uniform(200, 2000),
                total_execution_time_ms=random.uniform(300, 2500),
                market_volatility=random.uniform(0.005, 0.05),
                spread_bps=random.uniform(1, 20),
                volume_at_signal=random.uniform(1000, 15000),
                result=random.choice([TradeResult.WIN, TradeResult.LOSS, TradeResult.BREAKEVEN]),
                strategy_context={
                    "market_regime": random.choice(["trending", "ranging", "volatile"]),
                    "time_of_day": timestamp.hour,
                    "confidence_threshold": 0.6,
                    "volatility_filter": True
                }
            )
            
            # 取引ログ記録
            await self.performance_logger.log_trade_execution(trade_execution)
            
            if (i + 1) % 10 == 0:
                print(f"   ✅ Processed {i+1}/{num_trades} trades")
        
        print(f"📊 Trading session completed: {num_trades} trades simulated")
        
    async def run_pdca_cycle_demo(self):
        """PDCAサイクルデモ実行"""
        print("\n🔄 Running PDCA Cycle Demo...")
        
        strategy_name = "EnhancedWeekendEffectStrategy"
        
        # Plan フェーズ
        print("📋 PLAN Phase: Setting hypothesis and targets")
        hypothesis = "信頼度閾値を0.7に上げることで勝率向上とリスク削減を実現"
        target_metrics = {
            "win_rate": 0.65,
            "profit_factor": 2.0,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08
        }
        parameter_changes = {
            "confidence_threshold": 0.7,
            "position_size_multiplier": 0.8
        }
        
        session_id = await self.performance_logger.start_pdca_cycle(
            strategy_name, hypothesis, target_metrics, parameter_changes
        )
        print(f"   📝 PDCA Session started: {session_id}")
        
        # Do フェーズ
        print("⚡ DO Phase: Logging execution")
        execution_summary = {
            "trades_executed": 50,
            "execution_period": "2025-06-01 to 2025-06-15",
            "parameter_adjustments_applied": parameter_changes,
            "market_conditions_encountered": ["trending", "ranging", "volatile"]
        }
        market_conditions = {
            "avg_volatility": 0.025,
            "dominant_regime": "trending",
            "significant_events": ["Federal Reserve meeting", "Employment data release"],
            "market_hours_active": 360  # 15 days * 24 hours
        }
        
        await self.performance_logger.log_execution_phase(strategy_name, execution_summary, market_conditions)
        print("   📈 Execution phase logged")
        
        # Check フェーズ
        print("🔍 CHECK Phase: Running comprehensive analysis")
        analysis_results = await self.performance_logger.analyze_performance(strategy_name)
        print("   📊 Performance analysis completed")
        
        # Act フェーズ
        print("💡 ACT Phase: Generating improvements")
        improvements = await self.performance_logger.generate_improvements(strategy_name, analysis_results)
        print("   🎯 Improvement recommendations generated")
        
        return {
            "session_id": session_id,
            "analysis": analysis_results,
            "improvements": improvements
        }
        
    async def run_comprehensive_analysis(self):
        """包括的分析実行"""
        print("\n🔬 Running Comprehensive Analysis...")
        
        strategy_name = "EnhancedWeekendEffectStrategy"
        analysis = await self.pdca_analyzer.run_comprehensive_analysis(strategy_name)
        
        print("📈 Analysis Results:")
        print(f"   Overall Score: {analysis['overall_score']:.1f}/100")
        
        # 基本パフォーマンス
        basic = analysis["analyses"]["basic_performance"]
        if "error" not in basic:
            print(f"   Win Rate: {basic['win_rate']:.1%}")
            print(f"   Profit Factor: {basic['profit_factor']:.2f}")
            print(f"   Sharpe Ratio: {basic['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {basic['max_drawdown']:.1%}")
        
        # 推奨事項
        recommendations = analysis["recommendations"]
        if recommendations:
            print("\n💡 Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec['title']}")
                print(f"      {rec['description']}")
                print(f"      Expected: {rec['expected_improvement']}")
        
        return analysis
    
    async def generate_performance_report(self):
        """パフォーマンスレポート生成"""
        print("\n📋 Generating Performance Report...")
        
        # リアルタイムダッシュボードデータ
        dashboard_data = await self.performance_logger.get_realtime_dashboard_data()
        
        # 戦略別レポートエクスポート
        strategy_report_path = await self.performance_logger.export_analysis_report("EnhancedWeekendEffectStrategy")
        
        # フルレポートエクスポート
        full_report_path = await self.performance_logger.export_analysis_report()
        
        print(f"📄 Strategy Report: {strategy_report_path}")
        print(f"📄 Full Report: {full_report_path}")
        
        # サマリー表示
        print("\n📊 Session Summary:")
        print(f"   Total Trades: {dashboard_data['overall_stats']['total_trades']}")
        print(f"   Active Strategies: {dashboard_data['overall_stats']['total_strategies']}")
        print(f"   Session Uptime: {dashboard_data['session_info']['uptime_minutes']:.1f} minutes")
        
        if dashboard_data["strategy_performance"]:
            for strategy, perf in dashboard_data["strategy_performance"].items():
                print(f"   {strategy}:")
                print(f"     Win Rate: {perf['win_rate']:.1%}")
                print(f"     Total P&L: ${perf['total_pnl']:.2f}")
                print(f"     Profit Factor: {perf['profit_factor']:.2f}")
        
        return {
            "dashboard_data": dashboard_data,
            "strategy_report_path": strategy_report_path,
            "full_report_path": full_report_path
        }
    
    async def run_complete_demo(self):
        """完全デモ実行"""
        print("=" * 80)
        print("🎯 ProjectChimera Performance Measurement System Demo")
        print("   取引ロジック完成度のPDCAサイクル管理システム")
        print("=" * 80)
        
        try:
            # 1. システム初期化
            await self.initialize()
            
            # 2. 取引データシミュレーション
            await self.simulate_trading_session(50)
            
            # 3. PDCAサイクル実行
            pdca_results = await self.run_pdca_cycle_demo()
            
            # 4. 包括的分析
            analysis_results = await self.run_comprehensive_analysis()
            
            # 5. レポート生成
            report_results = await self.generate_performance_report()
            
            # 6. デモ完了サマリー
            print("\n" + "=" * 80)
            print("✅ Demo Completed Successfully!")
            print("=" * 80)
            print("🎯 Key Features Demonstrated:")
            print("   ✅ リアルタイム取引パフォーマンス追跡")
            print("   ✅ 戦略別詳細分析（時系列・レジーム・リスク）")
            print("   ✅ PDCAサイクル管理（Plan→Do→Check→Act）")
            print("   ✅ 自動最適化提案生成")
            print("   ✅ 包括的レポート出力")
            print("   ✅ リアルタイムダッシュボード対応")
            
            print("\n📊 Next Steps:")
            print("   1. Run: python run_performance_dashboard.py")
            print("   2. Open: http://localhost:8502")
            print("   3. Explore real-time performance dashboard")
            print("   4. Review generated reports in logs/demo_performance/")
            
            return {
                "status": "success",
                "pdca_results": pdca_results,
                "analysis_results": analysis_results,
                "report_results": report_results
            }
            
        except Exception as e:
            print(f"\n❌ Demo failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}


async def main():
    """メイン実行関数"""
    demo = PerformanceSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())