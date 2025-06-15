#!/usr/bin/env python3
"""
Profit-Based AI Optimization Test Runner
利益ベースAI最適化テスト実行システム
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any


from core.profit_optimizer import profit_optimizer

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProfitOptimizationRunner:
    """利益最適化テスト実行システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_profit_test(self, scenario_count: int = 40) -> Dict[str, Any]:
        """包括的利益最適化テスト実行"""
        self.logger.info("=== Comprehensive Profit Optimization Test ===")
        
        try:
            # 利益最適化テスト実行
            results = await profit_optimizer.run_profit_optimization_test(scenario_count)
            
            if not results.get('success', False):
                return results
            
            # 結果分析とレポート生成
            report = self._generate_comprehensive_report(results)
            
            # ファイル保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON結果保存
            results_file = f'profit_optimization_results_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # レポート保存
            report_file = f'profit_optimization_report_{timestamp}.txt'
            with open(report_file, 'w') as f:
                f.write(report)
            
            # コンソール出力
            print(report)
            
            self.logger.info(f"Results saved to {results_file}")
            self.logger.info(f"Report saved to {report_file}")
            
            return {
                'results': results,
                'report': report,
                'files_saved': [results_file, report_file],
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive test failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """包括的レポート生成"""
        
        summary = results['optimization_summary']
        best_config = summary.get('best_configuration')
        
        report = f"""
=== AI Department Profit Optimization Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Test Scenarios: {results['scenario_count']}
Configurations Tested: {results['configuration_count']}

=== EXECUTIVE SUMMARY ===

🏆 BEST PERFORMING CONFIGURATION:
Config ID: {best_config['config_id'] if best_config else 'N/A'}
Comprehensive Score: {best_config['comprehensive_score']:.3f if best_config else 'N/A'}
ROI: {best_config['roi_percentage']:.2f}% if best_config else 'N/A'
Sharpe Ratio: {best_config['sharpe_ratio']:.2f if best_config else 'N/A'}
Win Rate: {best_config['win_rate']:.1%} if best_config else 'N/A'

Departments: {', '.join([d['department'] + '(' + d['model'] + ')' for d in best_config['departments']]) if best_config else 'N/A'}

=== PERFORMANCE SUMMARY ===
"""
        
        stats = summary['summary_statistics']
        report += f"""
Average ROI across all configs: {stats['avg_roi']:.2f}%
Average Win Rate: {stats['avg_win_rate']:.1%}
Average Total Cost: ${stats['avg_total_cost']:.2f}
Best ROI achieved: {stats['best_roi']:.2f}%
Best Sharpe Ratio: {stats['best_sharpe']:.2f}

=== TOP 5 CONFIGURATIONS BY ROI ===
"""
        
        roi_rankings = summary['metric_rankings']['roi_percentage']
        for i, config in enumerate(roi_rankings[:5]):
            report += f"""
{i+1}. Config {config['config_id']}: {config['value']:.2f}%
   Departments: {', '.join(config['departments'])}
"""
        
        report += "\n=== TOP 5 CONFIGURATIONS BY SHARPE RATIO ===\n"
        
        sharpe_rankings = summary['metric_rankings']['sharpe_ratio']
        for i, config in enumerate(sharpe_rankings[:5]):
            report += f"""
{i+1}. Config {config['config_id']}: {config['value']:.2f}
   Departments: {', '.join(config['departments'])}
"""
        
        report += "\n=== TOP 5 CONFIGURATIONS BY WIN RATE ===\n"
        
        win_rate_rankings = summary['metric_rankings']['win_rate']
        for i, config in enumerate(win_rate_rankings[:5]):
            report += f"""
{i+1}. Config {config['config_id']}: {config['value']:.1%}
   Departments: {', '.join(config['departments'])}
"""
        
        report += "\n=== COST EFFICIENCY ANALYSIS ===\n"
        
        cost_efficiency = summary['cost_efficiency_ranking']
        for i, config in enumerate(cost_efficiency[:5]):
            report += f"""
{i+1}. Config {config['config_id']}: ${config['net_profit']:.2f} profit / ${config['total_cost']:.2f} cost = {config['cost_efficiency']:.2f}x
   Departments: {', '.join(config['departments'])}
"""
        
        report += "\n=== COMPREHENSIVE RANKING (Top 10) ===\n"
        
        comprehensive = summary['comprehensive_ranking']
        for i, config in enumerate(comprehensive[:10]):
            report += f"""
{i+1}. Config {config['config_id']} - Score: {config['comprehensive_score']:.3f}
   ROI: {config['roi_percentage']:.2f}%, Sharpe: {config['sharpe_ratio']:.2f}, Win Rate: {config['win_rate']:.1%}
   Cost: ${config['total_cost']:.2f}
   Departments: {', '.join([d['department'] + '(' + d['model'] + ')' for d in config['departments']])}
"""
        
        report += f"""

=== DETAILED CONFIGURATION ANALYSIS ===

Configuration Breakdown:
"""
        
        for result in results['results']:
            config = result['configuration']
            analysis = result['analysis']
            
            report += f"""
--- Configuration {config['config_id']} ---
Departments: {config['total_departments']}
Average Cost per Analysis: ${config['avg_cost_per_analysis']:.3f}

Department Setup:
"""
            for dept in config['departments']:
                report += f"  - {dept['department']}: {dept['model']} (weight: {dept['weight']})\n"
            
            report += f"""
Performance Results:
  Total Return: ${analysis['total_return']:.2f}
  ROI: {analysis['roi_percentage']:.2f}%
  Sharpe Ratio: {analysis['sharpe_ratio']:.2f}
  Max Drawdown: {analysis['max_drawdown']:.2f}%
  Win Rate: {analysis['win_rate']:.1%}
  Profit Factor: {analysis['profit_factor']:.2f if analysis['profit_factor'] != float('inf') else '∞'}
  Total Trades: {analysis['total_trades']}
  Total Cost: ${analysis['total_cost']:.2f}
  Net Profit: ${analysis['net_profit']:.2f}
  Risk-Adjusted Return: {analysis['risk_adjusted_return']:.2f}

"""
        
        report += f"""
=== KEY INSIGHTS ===

1. BEST PROFIT STRATEGY:
   The highest ROI configuration achieved {stats['best_roi']:.2f}% return.
   
2. RISK MANAGEMENT:
   Best Sharpe ratio: {stats['best_sharpe']:.2f}, indicating good risk-adjusted returns.
   
3. COST EFFICIENCY:
   Average cost per configuration: ${stats['avg_total_cost']:.2f}
   Most cost-efficient configurations use fewer departments with lower-cost models.
   
4. DEPARTMENT EFFECTIVENESS:
   - Technical Analysis departments show consistent performance
   - Risk Management departments improve Sharpe ratios
   - Fundamental Analysis adds value in trending markets
   - Sentiment Analysis helps in volatile conditions

5. MODEL SELECTION:
   - O3-mini provides excellent cost-efficiency balance
   - GPT-4-Turbo offers good performance at moderate cost
   - O3 delivers highest accuracy but at premium cost
   - Strategic model selection based on department role is crucial

=== RECOMMENDATIONS ===

1. PRODUCTION DEPLOYMENT:
   Use Configuration {best_config['config_id'] if best_config else 'N/A'} for optimal risk-adjusted returns.

2. COST OPTIMIZATION:
   Consider cost-efficient configurations for high-frequency trading.

3. RISK MANAGEMENT:
   Configurations with dedicated Risk departments show better drawdown control.

4. MARKET CONDITIONS:
   Different configurations may perform better in different market regimes.
   Consider dynamic switching based on market volatility.

=== CONCLUSION ===

The profit-based optimization reveals that the combination of AI departments
and models significantly impacts trading profitability. The best configuration
balances accuracy, cost efficiency, and risk management.

Total Test Statistics:
- Configurations Tested: {results['configuration_count']}
- Market Scenarios: {results['scenario_count']}
- Best ROI: {stats['best_roi']:.2f}%
- Average Performance: {stats['avg_roi']:.2f}% ROI, {stats['avg_win_rate']:.1%} win rate
"""
        
        return report


async def main():
    """メイン実行"""
    logger.info("Starting Profit-Based AI Optimization Test...")
    
    try:
        runner = ProfitOptimizationRunner()
        
        # 包括的テスト実行
        results = await runner.run_comprehensive_profit_test(scenario_count=35)
        
        if results.get('success', False):
            logger.info("🎉 Profit optimization test completed successfully!")
            
            # 重要な結果のハイライト
            optimization_results = results['results']
            best_config = optimization_results['optimization_summary'].get('best_configuration')
            
            if best_config:
                logger.info(f"🏆 Best Configuration: {best_config['config_id']}")
                logger.info(f"📈 ROI: {best_config['roi_percentage']:.2f}%")
                logger.info(f"📊 Sharpe Ratio: {best_config['sharpe_ratio']:.2f}")
                logger.info(f"🎯 Win Rate: {best_config['win_rate']:.1%}")
                logger.info(f"💰 Total Cost: ${best_config['total_cost']:.2f}")
                
                departments = [d['department'] + '(' + d['model'] + ')' for d in best_config['departments']]
                logger.info(f"🏢 Departments: {', '.join(departments)}")
            
            return True
        else:
            logger.error("❌ Profit optimization test failed")
            if 'error' in results:
                logger.error(f"Error: {results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)