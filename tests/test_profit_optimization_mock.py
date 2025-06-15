#!/usr/bin/env python3
"""
Mock Profit-Based AI Optimization Test
モック利益ベースAI最適化テスト（API不要版）
"""

import asyncio
import json
import logging
import random
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockDepartmentType:
    """モック部門タイプ"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    RISK = "risk"
    EXECUTION = "execution"


class MockProfitOptimizer:
    """モック利益最適化システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 取引パラメータ
        self.base_position_size = 10000  # $10,000
        self.spread_cost = 0.0002  # 0.02%
        self.commission_rate = 0.0001  # 0.01%
        self.max_risk_per_trade = 0.02  # 2%
        
        # モデル設定
        self.model_configs = {
            "gpt-4": {"accuracy": 0.72, "cost": 0.08},
            "gpt-4-turbo": {"accuracy": 0.75, "cost": 0.04},
            "o3-mini": {"accuracy": 0.78, "cost": 0.03},
            "o3": {"accuracy": 0.82, "cost": 0.15}
        }
        
        # 部門別精度調整
        self.department_multipliers = {
            MockDepartmentType.TECHNICAL: 1.0,
            MockDepartmentType.FUNDAMENTAL: 0.9,
            MockDepartmentType.SENTIMENT: 0.85,
            MockDepartmentType.RISK: 1.1,
            MockDepartmentType.EXECUTION: 1.05
        }
    
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """部門設定パターン生成"""
        return [
            # パターン1: 全部門フル活用
            {
                "id": 1,
                "name": "Full Department Suite",
                "departments": [
                    {"type": MockDepartmentType.TECHNICAL, "model": "o3-mini", "weight": 0.35, "cost": 0.03},
                    {"type": MockDepartmentType.FUNDAMENTAL, "model": "gpt-4", "weight": 0.25, "cost": 0.08},
                    {"type": MockDepartmentType.SENTIMENT, "model": "o3-mini", "weight": 0.20, "cost": 0.03},
                    {"type": MockDepartmentType.RISK, "model": "o3", "weight": 0.15, "cost": 0.15},
                    {"type": MockDepartmentType.EXECUTION, "model": "o3-mini", "weight": 0.05, "cost": 0.03}
                ]
            },
            
            # パターン2: コスト効率重視
            {
                "id": 2,
                "name": "Cost Efficient",
                "departments": [
                    {"type": MockDepartmentType.TECHNICAL, "model": "o3-mini", "weight": 0.45, "cost": 0.03},
                    {"type": MockDepartmentType.FUNDAMENTAL, "model": "o3-mini", "weight": 0.25, "cost": 0.03},
                    {"type": MockDepartmentType.SENTIMENT, "model": "gpt-4-turbo", "weight": 0.20, "cost": 0.04},
                    {"type": MockDepartmentType.RISK, "model": "gpt-4", "weight": 0.10, "cost": 0.08}
                ]
            },
            
            # パターン3: テクニカル重視
            {
                "id": 3,
                "name": "Technical Focused",
                "departments": [
                    {"type": MockDepartmentType.TECHNICAL, "model": "o3", "weight": 0.60, "cost": 0.15},
                    {"type": MockDepartmentType.RISK, "model": "o3", "weight": 0.25, "cost": 0.15},
                    {"type": MockDepartmentType.EXECUTION, "model": "o3-mini", "weight": 0.15, "cost": 0.03}
                ]
            },
            
            # パターン4: ファンダメンタル重視
            {
                "id": 4,
                "name": "Fundamental Focused", 
                "departments": [
                    {"type": MockDepartmentType.FUNDAMENTAL, "model": "gpt-4", "weight": 0.50, "cost": 0.08},
                    {"type": MockDepartmentType.SENTIMENT, "model": "o3-mini", "weight": 0.30, "cost": 0.03},
                    {"type": MockDepartmentType.RISK, "model": "gpt-4", "weight": 0.20, "cost": 0.08}
                ]
            },
            
            # パターン5: バランス型
            {
                "id": 5,
                "name": "Balanced",
                "departments": [
                    {"type": MockDepartmentType.TECHNICAL, "model": "o3-mini", "weight": 0.30, "cost": 0.03},
                    {"type": MockDepartmentType.FUNDAMENTAL, "model": "o3-mini", "weight": 0.25, "cost": 0.03},
                    {"type": MockDepartmentType.SENTIMENT, "model": "gpt-4-turbo", "weight": 0.25, "cost": 0.04},
                    {"type": MockDepartmentType.RISK, "model": "gpt-4-turbo", "weight": 0.20, "cost": 0.04}
                ]
            },
            
            # パターン6: 超低コスト
            {
                "id": 6,
                "name": "Ultra Low Cost",
                "departments": [
                    {"type": MockDepartmentType.TECHNICAL, "model": "o3-mini", "weight": 0.70, "cost": 0.03},
                    {"type": MockDepartmentType.RISK, "model": "o3-mini", "weight": 0.30, "cost": 0.03}
                ]
            }
        ]
    
    def generate_market_scenarios(self, count: int = 40) -> List[Dict[str, Any]]:
        """市場シナリオ生成"""
        scenarios = []
        
        scenario_types = [
            ("strong_bullish", 0.15, 0.025),    # 強いブル: 15%, 2.5%上昇
            ("moderate_bullish", 0.25, 0.015),  # 中程度ブル: 25%, 1.5%上昇
            ("neutral", 0.30, 0.005),           # 中立: 30%, 0.5%変動
            ("moderate_bearish", 0.20, -0.012), # 中程度ベア: 20%, 1.2%下落
            ("strong_bearish", 0.10, -0.020)    # 強いベア: 10%, 2.0%下落
        ]
        
        for i in range(count):
            scenario_type, probability, expected_return = random.choices(
                scenario_types, weights=[s[1] for s in scenario_types]
            )[0]
            
            scenario = {
                "id": i + 1,
                "type": scenario_type,
                "expected_return": expected_return,
                "volatility": random.uniform(0.005, 0.025),
                "difficulty": random.uniform(0.3, 0.9)
            }
            scenarios.append(scenario)
        
        return scenarios
    
    async def simulate_trading_session(
        self, 
        config: Dict[str, Any], 
        scenarios: List[Dict[str, Any]], 
        initial_balance: float = 100000
    ) -> Dict[str, Any]:
        """取引セッションシミュレーション"""
        
        session = {
            "config_id": config["id"],
            "config_name": config["name"],
            "start_balance": initial_balance,
            "current_balance": initial_balance,
            "trades": [],
            "total_ai_cost": 0.0,
            "start_time": datetime.now()
        }
        
        for scenario in scenarios:
            trade_result = await self._simulate_trade(config, scenario, session["current_balance"])
            
            if trade_result:
                session["trades"].append(trade_result)
                session["current_balance"] += trade_result["net_profit_loss"]
                session["total_ai_cost"] += trade_result["ai_cost"]
                
                # ドローダウンチェック
                drawdown = (session["start_balance"] - session["current_balance"]) / session["start_balance"]
                if drawdown > 0.05:  # 5%以上のドローダウンで停止
                    break
        
        session["end_time"] = datetime.now()
        return session
    
    async def _simulate_trade(
        self, 
        config: Dict[str, Any], 
        scenario: Dict[str, Any], 
        current_balance: float
    ) -> Dict[str, Any]:
        """個別取引シミュレーション"""
        
        # AI部門分析
        department_decisions = []
        total_ai_cost = 0.0
        
        for dept in config["departments"]:
            decision = self._simulate_department_decision(dept, scenario)
            department_decisions.append(decision)
            total_ai_cost += dept["cost"]
        
        # 統合判定
        final_decision = self._integrate_decisions(department_decisions, config)
        
        if not final_decision["trade_warranted"]:
            return {
                "scenario_id": scenario["id"],
                "action": "hold",
                "ai_cost": total_ai_cost,
                "net_profit_loss": -total_ai_cost,
                "details": final_decision
            }
        
        # 取引実行シミュレーション
        return await self._execute_trade(final_decision, scenario, total_ai_cost, current_balance)
    
    def _simulate_department_decision(self, dept: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """部門判定シミュレーション"""
        
        model_config = self.model_configs[dept["model"]]
        dept_multiplier = self.department_multipliers[dept["type"]]
        
        # 成功確率計算
        base_accuracy = model_config["accuracy"] * dept_multiplier
        difficulty_penalty = scenario["difficulty"]
        success_probability = base_accuracy * (1.2 - difficulty_penalty)
        success_probability = max(0.1, min(0.95, success_probability))
        
        is_correct = random.random() < success_probability
        confidence = random.uniform(0.6, 0.95) if is_correct else random.uniform(0.3, 0.7)
        
        # 方向決定
        expected_return = scenario["expected_return"]
        if is_correct:
            if expected_return > 0.01:
                direction = "long"
            elif expected_return < -0.01:
                direction = "short"
            else:
                direction = "hold"
        else:
            direction = random.choice(["long", "short", "hold"])
        
        return {
            "department": dept["type"],
            "model": dept["model"],
            "direction": direction,
            "confidence": confidence,
            "weight": dept["weight"],
            "trade_warranted": direction != "hold" and confidence > 0.6,
            "is_correct": is_correct
        }
    
    def _integrate_decisions(self, decisions: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """部門判定統合"""
        
        # 重み付き投票
        total_weight = sum(d["weight"] for d in decisions)
        weighted_confidence = sum(d["confidence"] * d["weight"] for d in decisions) / total_weight
        
        # 方向決定
        direction_votes = {"long": 0, "short": 0, "hold": 0}
        for decision in decisions:
            direction_votes[decision["direction"]] += decision["weight"]
        
        final_direction = max(direction_votes, key=direction_votes.get)
        
        # 取引実行判定
        trade_votes = sum(d["weight"] for d in decisions if d["trade_warranted"])
        trade_warranted = (
            trade_votes > total_weight * 0.6 and
            weighted_confidence > 0.65 and
            final_direction != "hold"
        )
        
        return {
            "trade_warranted": trade_warranted,
            "direction": final_direction,
            "confidence": weighted_confidence,
            "consensus_strength": direction_votes[final_direction] / total_weight
        }
    
    async def _execute_trade(
        self, 
        decision: Dict[str, Any], 
        scenario: Dict[str, Any], 
        ai_cost: float, 
        current_balance: float
    ) -> Dict[str, Any]:
        """取引実行シミュレーション"""
        
        direction = decision["direction"]
        confidence = decision["confidence"]
        
        # ポジションサイズ決定
        risk_amount = current_balance * self.max_risk_per_trade
        position_size = risk_amount * confidence
        position_size = min(position_size, self.base_position_size)
        
        # 市場結果
        actual_return = scenario["expected_return"] + random.gauss(0, scenario["volatility"])
        actual_return = max(-0.05, min(0.05, actual_return))  # -5%~+5%制限
        
        # P&L計算
        if direction == "long":
            gross_profit_loss = position_size * actual_return
        elif direction == "short":
            gross_profit_loss = position_size * (-actual_return)
        else:
            gross_profit_loss = 0.0
        
        # コスト計算
        trading_cost = position_size * (self.spread_cost + self.commission_rate)
        net_profit_loss = gross_profit_loss - trading_cost - ai_cost
        
        return {
            "scenario_id": scenario["id"],
            "action": direction,
            "position_size": position_size,
            "actual_return": actual_return,
            "gross_profit_loss": gross_profit_loss,
            "trading_cost": trading_cost,
            "ai_cost": ai_cost,
            "net_profit_loss": net_profit_loss,
            "confidence": confidence
        }
    
    def analyze_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """セッション分析"""
        
        if not session["trades"]:
            return {
                "total_return": -session["total_ai_cost"],
                "roi_percentage": -100.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 100.0,
                "sharpe_ratio": 0.0,
                "total_trades": 0,
                "total_cost": session["total_ai_cost"]
            }
        
        trades = session["trades"]
        start_balance = session["start_balance"]
        
        # 基本統計
        total_return = session["current_balance"] - start_balance
        roi_percentage = (total_return / start_balance) * 100
        
        profitable_trades = [t for t in trades if t["net_profit_loss"] > 0]
        win_rate = len(profitable_trades) / len(trades)
        
        # プロフィットファクター
        gross_profits = sum(t["net_profit_loss"] for t in trades if t["net_profit_loss"] > 0)
        gross_losses = abs(sum(t["net_profit_loss"] for t in trades if t["net_profit_loss"] < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # 最大ドローダウン
        running_balance = start_balance
        peak_balance = start_balance
        max_drawdown = 0.0
        
        for trade in trades:
            running_balance += trade["net_profit_loss"]
            peak_balance = max(peak_balance, running_balance)
            if peak_balance > 0:
                drawdown = (peak_balance - running_balance) / peak_balance
                max_drawdown = max(max_drawdown, drawdown)
        
        # シャープレシオ
        if len(trades) > 1:
            returns = [t["net_profit_loss"] / start_balance for t in trades]
            mean_return = sum(returns) / len(returns)
            return_std = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = (mean_return * 252) / (return_std * (252 ** 0.5)) if return_std > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        return {
            "total_return": total_return,
            "roi_percentage": roi_percentage,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown * 100,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(trades),
            "profitable_trades": len(profitable_trades),
            "total_cost": session["total_ai_cost"],
            "net_profit": total_return - session["total_ai_cost"]
        }


class ProfitOptimizationRunner:
    """利益最適化テスト実行"""
    
    def __init__(self):
        self.optimizer = MockProfitOptimizer()
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_test(self, scenario_count: int = 35) -> Dict[str, Any]:
        """包括的テスト実行"""
        
        self.logger.info(f"Starting profit optimization test with {scenario_count} scenarios")
        
        # 設定とシナリオ生成
        configurations = self.optimizer.generate_configurations()
        scenarios = self.optimizer.generate_market_scenarios(scenario_count)
        
        results = []
        
        for config in configurations:
            self.logger.info(f"Testing configuration {config['id']}: {config['name']}")
            
            # 取引セッション実行
            session = await self.optimizer.simulate_trading_session(config, scenarios)
            
            # 分析
            analysis = self.optimizer.analyze_session(session)
            
            result = {
                "configuration": config,
                "session": session,
                "analysis": analysis
            }
            results.append(result)
        
        # 最適化分析
        optimization_summary = self._analyze_optimization_results(results)
        
        return {
            "test_timestamp": datetime.now().isoformat(),
            "scenario_count": scenario_count,
            "configuration_count": len(configurations),
            "results": results,
            "optimization_summary": optimization_summary
        }
    
    def _analyze_optimization_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """最適化結果分析"""
        
        # ROIランキング
        roi_ranking = sorted(results, key=lambda x: x["analysis"]["roi_percentage"], reverse=True)
        
        # シャープレシオランキング
        sharpe_ranking = sorted(
            results, 
            key=lambda x: x["analysis"]["sharpe_ratio"] if x["analysis"]["sharpe_ratio"] != float('inf') else -999, 
            reverse=True
        )
        
        # 勝率ランキング
        win_rate_ranking = sorted(results, key=lambda x: x["analysis"]["win_rate"], reverse=True)
        
        # コスト効率ランキング
        cost_efficiency = []
        for result in results:
            analysis = result["analysis"]
            if analysis["total_cost"] > 0:
                efficiency = analysis["net_profit"] / analysis["total_cost"]
            else:
                efficiency = analysis["net_profit"]
            cost_efficiency.append({
                "config": result["configuration"],
                "efficiency": efficiency,
                "analysis": analysis
            })
        
        cost_efficiency.sort(key=lambda x: x["efficiency"], reverse=True)
        
        # 総合スコア
        comprehensive_scores = []
        for result in results:
            analysis = result["analysis"]
            
            roi_score = max(0, analysis["roi_percentage"] / 100)
            sharpe_score = max(0, min(1, analysis["sharpe_ratio"] / 3))
            win_rate_score = analysis["win_rate"]
            
            comprehensive_score = roi_score * 0.4 + sharpe_score * 0.3 + win_rate_score * 0.3
            
            comprehensive_scores.append({
                "config": result["configuration"],
                "score": comprehensive_score,
                "analysis": analysis
            })
        
        comprehensive_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "roi_ranking": roi_ranking[:5],
            "sharpe_ranking": sharpe_ranking[:5],
            "win_rate_ranking": win_rate_ranking[:5],
            "cost_efficiency_ranking": cost_efficiency[:5],
            "comprehensive_ranking": comprehensive_scores,
            "best_configuration": comprehensive_scores[0] if comprehensive_scores else None,
            "summary_stats": {
                "avg_roi": sum(r["analysis"]["roi_percentage"] for r in results) / len(results),
                "avg_win_rate": sum(r["analysis"]["win_rate"] for r in results) / len(results),
                "avg_sharpe": sum(r["analysis"]["sharpe_ratio"] for r in results if r["analysis"]["sharpe_ratio"] != float('inf')) / len(results),
                "best_roi": max(r["analysis"]["roi_percentage"] for r in results),
                "best_sharpe": max(r["analysis"]["sharpe_ratio"] for r in results if r["analysis"]["sharpe_ratio"] != float('inf'))
            }
        }
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """レポート生成"""
        
        summary = test_results["optimization_summary"]
        best_config = summary["best_configuration"]
        
        report = f"""
=== PROFIT-BASED AI OPTIMIZATION REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Test Scenarios: {test_results['scenario_count']}
Configurations Tested: {test_results['configuration_count']}

=== BEST PERFORMING CONFIGURATION ===
🏆 Configuration {best_config['config']['id']}: {best_config['config']['name']}
📈 ROI: {best_config['analysis']['roi_percentage']:.2f}%
📊 Sharpe Ratio: {best_config['analysis']['sharpe_ratio']:.2f}
🎯 Win Rate: {best_config['analysis']['win_rate']:.1%}
💰 Net Profit: ${best_config['analysis']['net_profit']:.2f}
💸 Total Cost: ${best_config['analysis']['total_cost']:.2f}
🔻 Max Drawdown: {best_config['analysis']['max_drawdown']:.2f}%

Department Configuration:
"""
        
        for dept in best_config['config']['departments']:
            report += f"  - {dept['type']}: {dept['model']} (weight: {dept['weight']}, cost: ${dept['cost']})\n"
        
        report += f"""

=== PERFORMANCE RANKINGS ===

TOP 3 BY ROI:
"""
        for i, result in enumerate(summary['roi_ranking'][:3]):
            analysis = result['analysis']
            config = result['configuration']
            report += f"{i+1}. {config['name']}: {analysis['roi_percentage']:.2f}% ROI\n"
        
        report += "\nTOP 3 BY SHARPE RATIO:\n"
        for i, result in enumerate(summary['sharpe_ranking'][:3]):
            analysis = result['analysis']
            config = result['configuration']
            sharpe = f"{analysis['sharpe_ratio']:.2f}" if analysis['sharpe_ratio'] != float('inf') else 'INF'
            report += f"{i+1}. {config['name']}: {sharpe} Sharpe Ratio\n"
        
        report += "\nTOP 3 BY WIN RATE:\n"
        for i, result in enumerate(summary['win_rate_ranking'][:3]):
            analysis = result['analysis']
            config = result['configuration']
            report += f"{i+1}. {config['name']}: {analysis['win_rate']:.1%} Win Rate\n"
        
        report += "\nTOP 3 BY COST EFFICIENCY:\n"
        for i, item in enumerate(summary['cost_efficiency_ranking'][:3]):
            config = item['config']
            efficiency = item['efficiency']
            report += f"{i+1}. {config['name']}: {efficiency:.2f}x Cost Efficiency\n"
        
        stats = summary['summary_stats']
        report += f"""

=== SUMMARY STATISTICS ===
Average ROI: {stats['avg_roi']:.2f}%
Average Win Rate: {stats['avg_win_rate']:.1%}
Average Sharpe Ratio: {stats['avg_sharpe']:.2f}
Best ROI: {stats['best_roi']:.2f}%
Best Sharpe Ratio: {stats['best_sharpe']:.2f}

=== DETAILED RESULTS ===
"""
        
        for result in test_results['results']:
            config = result['configuration']
            analysis = result['analysis']
            
            report += f"""
--- Configuration {config['id']}: {config['name']} ---
Departments: {len(config['departments'])}
"""
            for dept in config['departments']:
                report += f"  {dept['type']}: {dept['model']} (weight {dept['weight']}, cost ${dept['cost']})\n"
            
            report += f"""
Results:
  ROI: {analysis['roi_percentage']:.2f}%
  Total Return: ${analysis['total_return']:.2f}
  Win Rate: {analysis['win_rate']:.1%}
  Sharpe Ratio: {f"{analysis['sharpe_ratio']:.2f}" if analysis['sharpe_ratio'] != float('inf') else "INF"}
  Max Drawdown: {analysis['max_drawdown']:.2f}%
  Total Trades: {analysis['total_trades']}
  Total Cost: ${analysis['total_cost']:.2f}
  Net Profit: ${analysis['net_profit']:.2f}

"""
        
        report += f"""
=== KEY INSIGHTS ===

1. HIGHEST PROFIT: Configuration {best_config['config']['id']} achieved {best_config['analysis']['roi_percentage']:.2f}% ROI

2. COST EFFICIENCY: {summary['cost_efficiency_ranking'][0]['config']['name']} provides best cost efficiency

3. RISK MANAGEMENT: Configurations with Risk departments show better drawdown control

4. DEPARTMENT SYNERGY: Multi-department setups generally outperform single-department configs

=== RECOMMENDATIONS ===

🚀 PRODUCTION: Use Configuration {best_config['config']['id']} for optimal profit
💰 COST-CONSCIOUS: Consider cost-efficient alternatives for high-frequency trading
⚖️ RISK-AWARE: Include Risk departments for better drawdown management
📊 MONITORING: Track performance across different market conditions

Total Test: {test_results['configuration_count']} configurations, {test_results['scenario_count']} scenarios
Best Performance: {best_config['analysis']['roi_percentage']:.2f}% ROI, {best_config['analysis']['win_rate']:.1%} win rate
"""
        
        return report


async def main():
    """メイン実行"""
    logger.info("=== Starting Mock Profit-Based AI Optimization Test ===")
    
    try:
        runner = ProfitOptimizationRunner()
        
        # 包括的テスト実行
        results = await runner.run_comprehensive_test(scenario_count=40)
        
        # レポート生成
        report = runner.generate_report(results)
        
        # ファイル保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_file = f'profit_optimization_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        report_file = f'profit_optimization_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # コンソール出力
        print(report)
        
        logger.info(f"🎉 Profit optimization test completed!")
        logger.info(f"📄 Results saved to {results_file}")
        logger.info(f"📊 Report saved to {report_file}")
        
        # ハイライト表示
        best_config = results["optimization_summary"]["best_configuration"]
        logger.info(f"🏆 Best Configuration: {best_config['config']['name']}")
        logger.info(f"📈 ROI: {best_config['analysis']['roi_percentage']:.2f}%")
        logger.info(f"🎯 Win Rate: {best_config['analysis']['win_rate']:.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)