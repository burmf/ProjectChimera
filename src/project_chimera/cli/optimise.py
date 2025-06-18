"""
Strategy Parameter Optimization with Optuna and Joblib
Phase E implementation - 50 trials with parallel processing
Target: Multi-strategy parameter optimization with comprehensive metrics
"""

import time
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import warnings
warnings.filterwarnings('ignore')

# Mock Optuna and joblib for systems without these packages
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not available, using mock optimization")
    OPTUNA_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    print("Joblib not available, using sequential processing")
    JOBLIB_AVAILABLE = False

# Import our backtesting engine
try:
    from .backtest import VectorizedBacktester, MarketData, BacktestConfig, BacktestResult, StrategyType
except ImportError:
    from project_chimera.cli.backtest import VectorizedBacktester, MarketData, BacktestConfig, BacktestResult, StrategyType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization"""
    n_trials: int = 50
    n_jobs: int = -1                    # -1 = use all cores
    timeout: Optional[float] = None     # Max time in seconds
    random_seed: int = 42
    optimization_direction: str = "maximize"  # maximize sharpe, minimize drawdown, etc.
    objective_metric: str = "sharpe_ratio"    # sharpe_ratio, calmar_ratio, total_return
    
    # Optuna specific
    study_name: Optional[str] = None
    storage: Optional[str] = None       # For distributed optimization
    load_if_exists: bool = True
    
    # Performance constraints
    min_trades: int = 10               # Minimum trades for valid result
    max_drawdown_limit: float = 0.5    # 50% max drawdown limit
    min_win_rate: float = 0.0          # Minimum win rate (0 = no limit)


@dataclass
class ParameterSpace:
    """Define parameter search spaces for strategies"""
    strategy_type: StrategyType
    parameters: Dict[str, Tuple[Union[int, float], Union[int, float]]]  # (min, max) tuples
    
    def sample_optuna(self, trial) -> Dict[str, Any]:
        """Sample parameters using Optuna trial"""
        params = {}
        for param_name, (min_val, max_val) in self.parameters.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            else:
                params[param_name] = trial.suggest_float(param_name, float(min_val), float(max_val))
        return params
    
    def sample_random(self, random_state=None) -> Dict[str, Any]:
        """Sample parameters randomly (fallback for no Optuna)"""
        import random
        if random_state:
            random.seed(random_state)
        
        params = {}
        for param_name, (min_val, max_val) in self.parameters.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = random.randint(min_val, max_val)
            else:
                params[param_name] = random.uniform(float(min_val), float(max_val))
        return params


@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    strategy_type: StrategyType
    best_params: Dict[str, Any]
    best_value: float
    best_metrics: Dict[str, Any]
    
    # Trial statistics
    n_trials: int
    completed_trials: int
    failed_trials: int
    
    # Performance data
    optimization_time: float
    trials_per_second: float
    
    # All trial results
    all_trials: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'strategy_type': self.strategy_type.value,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'best_metrics': self.best_metrics,
            'n_trials': self.n_trials,
            'completed_trials': self.completed_trials,
            'failed_trials': self.failed_trials,
            'optimization_time': self.optimization_time,
            'trials_per_second': self.trials_per_second,
            'trial_count': len(self.all_trials)
        }


class StrategyOptimizer:
    """
    Strategy parameter optimizer using Optuna and parallel processing
    
    Features:
    - Multi-objective optimization (Sharpe, Calmar, Return, etc.)
    - Parallel trial execution with joblib
    - Comprehensive parameter search spaces
    - Constraint handling (min trades, max drawdown)
    - Robust error handling and logging
    """
    
    def __init__(self, config: OptimizationConfig, data: MarketData):
        self.config = config
        self.data = data
        self.backtester = VectorizedBacktester(BacktestConfig())
        
        # Define parameter spaces for each strategy
        self.parameter_spaces = self._define_parameter_spaces()
        
        # Results storage
        self.optimization_results: Dict[StrategyType, OptimizationResult] = {}
    
    def _define_parameter_spaces(self) -> Dict[StrategyType, ParameterSpace]:
        """Define parameter search spaces for all strategies"""
        spaces = {
            StrategyType.VOL_BREAKOUT: ParameterSpace(
                StrategyType.VOL_BREAKOUT,
                {
                    'bb_window': (10, 50),
                    'squeeze_percentile': (10, 30),
                    'breakout_threshold': (0.01, 0.05),
                }
            ),
            
            StrategyType.MINI_MOMENTUM: ParameterSpace(
                StrategyType.MINI_MOMENTUM,
                {
                    'lookback_window': (3, 20),
                    'z_threshold': (0.5, 3.0),
                    'holding_period': (1, 10),
                }
            ),
            
            StrategyType.LOB_REVERT: ParameterSpace(
                StrategyType.LOB_REVERT,
                {
                    'rsi_window': (5, 30),
                    'oversold_threshold': (20, 40),
                    'overbought_threshold': (60, 80),
                }
            ),
            
            StrategyType.FUNDING_ALPHA: ParameterSpace(
                StrategyType.FUNDING_ALPHA,
                {
                    'momentum_window': (5, 25),
                    'momentum_threshold': (0.02, 0.10),
                    'mean_reversion_strength': (0.5, 2.0),
                }
            ),
            
            StrategyType.BASIS_ARB: ParameterSpace(
                StrategyType.BASIS_ARB,
                {
                    'vol_window': (10, 40),
                    'vol_percentile': (70, 95),
                    'position_hold_time': (1, 20),
                }
            ),
            
            StrategyType.CME_GAP: ParameterSpace(
                StrategyType.CME_GAP,
                {
                    'gap_threshold': (0.005, 0.03),
                    'fade_strength': (0.5, 2.0),
                    'max_hold_periods': (1, 10),
                }
            ),
            
            StrategyType.CASCADE_PRED: ParameterSpace(
                StrategyType.CASCADE_PRED,
                {
                    'cascade_window': (3, 15),
                    'cascade_threshold': (0.02, 0.08),
                    'buy_dip_strength': (0.5, 2.0),
                }
            ),
            
            StrategyType.ROUND_REV: ParameterSpace(
                StrategyType.ROUND_REV,
                {
                    'round_level_tolerance': (0.002, 0.02),
                    'fade_strength': (0.5, 2.0),
                    'max_distance_from_round': (0.001, 0.01),
                }
            ),
            
            StrategyType.STOP_REV: ParameterSpace(
                StrategyType.STOP_REV,
                {
                    'stop_hunt_window': (3, 10),
                    'stop_hunt_threshold': (0.02, 0.06),
                    'volume_spike_multiplier': (1.5, 5.0),
                }
            ),
            
            StrategyType.SESSION_BRK: ParameterSpace(
                StrategyType.SESSION_BRK,
                {
                    'range_window': (10, 50),
                    'breakout_strength': (1.0, 1.1),
                    'trend_follow_period': (1, 15),
                }
            ),
        }
        
        return spaces
    
    def objective_function(self, strategy_type: StrategyType, params: Dict[str, Any]) -> float:
        """
        Objective function for optimization
        Returns the metric to optimize (higher is better for maximization)
        """
        try:
            # Generate signals with parameters
            signals = self._generate_parameterized_signals(strategy_type, params)
            
            # Run backtest
            result = self.backtester._execute_vectorized_backtest(self.data, signals)
            
            # Apply constraints
            if result.total_trades < self.config.min_trades:
                return -1000  # Heavily penalize insufficient trades
            
            if abs(result.max_drawdown) > self.config.max_drawdown_limit:
                return -1000  # Heavily penalize excessive drawdown
            
            if result.win_rate < self.config.min_win_rate:
                return -1000  # Penalize low win rate if required
            
            # Return the target metric
            metric_value = getattr(result, self.config.objective_metric, 0.0)
            
            # Handle NaN/inf values
            if not np.isfinite(metric_value):
                return -1000
            
            return metric_value
            
        except Exception as e:
            logger.debug(f"Objective function failed: {e}")
            return -1000  # Return very bad score for failed trials
    
    def _generate_parameterized_signals(self, strategy_type: StrategyType, params: Dict[str, Any]) -> 'np.ndarray':
        """Generate signals with specific parameters"""
        import numpy as np
        
        # This is a simplified implementation
        # In practice, each strategy would use its specific parameters
        
        if strategy_type == StrategyType.VOL_BREAKOUT:
            return self._vol_breakout_with_params(params)
        elif strategy_type == StrategyType.MINI_MOMENTUM:
            return self._mini_momentum_with_params(params)
        elif strategy_type == StrategyType.LOB_REVERT:
            return self._lob_revert_with_params(params)
        else:
            # Default implementation for other strategies
            return self._default_strategy_with_params(strategy_type, params)
    
    def _vol_breakout_with_params(self, params: Dict[str, Any]) -> 'np.ndarray':
        """Parameterized volatility breakout strategy"""
        import numpy as np
        
        window = params.get('bb_window', 20)
        squeeze_percentile = params.get('squeeze_percentile', 20)
        breakout_threshold = params.get('breakout_threshold', 0.02)
        
        closes = self.data.closes
        
        # Bollinger Bands with custom parameters
        sma = np.convolve(closes, np.ones(window)/window, mode='valid')
        std = np.array([np.std(closes[i:i+window]) for i in range(len(closes)-window+1)])
        
        # Pad to match original length
        sma = np.concatenate([np.full(window-1, sma[0]), sma])
        std = np.concatenate([np.full(window-1, std[0]), std])
        
        bb_width = 2 * std / sma
        threshold = np.percentile(bb_width[window:], squeeze_percentile)
        
        signals = np.zeros(len(closes))
        
        for i in range(window, len(closes)):
            if bb_width[i] < threshold:  # Squeeze condition
                price_change = (closes[i] - closes[i-1]) / closes[i-1]
                if price_change > breakout_threshold:
                    signals[i] = 1
                elif price_change < -breakout_threshold:
                    signals[i] = -1
        
        return signals
    
    def _mini_momentum_with_params(self, params: Dict[str, Any]) -> 'np.ndarray':
        """Parameterized mini momentum strategy"""
        import numpy as np
        
        window = params.get('lookback_window', 7)
        z_threshold = params.get('z_threshold', 1.0)
        
        closes = self.data.closes
        signals = np.zeros(len(closes))
        
        for i in range(window, len(closes)):
            recent_prices = closes[i-window:i]
            z_score = (closes[i] - np.mean(recent_prices)) / (np.std(recent_prices) + 1e-8)
            
            if z_score > z_threshold:
                signals[i] = 1
            elif z_score < -z_threshold:
                signals[i] = -1
        
        return signals
    
    def _lob_revert_with_params(self, params: Dict[str, Any]) -> 'np.ndarray':
        """Parameterized LOB reversion strategy"""
        import numpy as np
        
        window = params.get('rsi_window', 14)
        oversold = params.get('oversold_threshold', 30)
        overbought = params.get('overbought_threshold', 70)
        
        closes = self.data.closes
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        signals = np.zeros(len(closes))
        
        for i in range(window, len(closes)):
            avg_gain = np.mean(gains[i-window:i])
            avg_loss = np.mean(losses[i-window:i])
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            if rsi < oversold:
                signals[i] = 1
            elif rsi > overbought:
                signals[i] = -1
        
        return signals
    
    def _default_strategy_with_params(self, strategy_type: StrategyType, params: Dict[str, Any]) -> 'np.ndarray':
        """Default parameterized strategy (simplified)"""
        import numpy as np
        
        # Simple momentum strategy with parameters
        window = params.get('window', 10)
        threshold = params.get('threshold', 0.02)
        
        closes = self.data.closes
        signals = np.zeros(len(closes))
        
        for i in range(window, len(closes)):
            momentum = (closes[i] - closes[i-window]) / closes[i-window]
            if momentum > threshold:
                signals[i] = 1
            elif momentum < -threshold:
                signals[i] = -1
        
        return signals
    
    def optimize_strategy(self, strategy_type: StrategyType) -> OptimizationResult:
        """Optimize parameters for a single strategy"""
        logger.info(f"Optimizing {strategy_type.value}...")
        start_time = time.time()
        
        if strategy_type not in self.parameter_spaces:
            raise ValueError(f"No parameter space defined for {strategy_type.value}")
        
        param_space = self.parameter_spaces[strategy_type]
        
        # Results storage
        all_trials = []
        best_value = float('-inf')
        best_params = {}
        best_metrics = {}
        
        if OPTUNA_AVAILABLE:
            # Use Optuna for optimization
            study = optuna.create_study(
                direction=self.config.optimization_direction,
                study_name=self.config.study_name,
                storage=self.config.storage,
                load_if_exists=self.config.load_if_exists
            )
            
            def optuna_objective(trial):
                params = param_space.sample_optuna(trial)
                return self.objective_function(strategy_type, params)
            
            study.optimize(
                optuna_objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=1  # Optuna handles parallelization
            )
            
            best_params = study.best_params
            best_value = study.best_value
            
            # Get all trial data
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    all_trials.append({
                        'params': trial.params,
                        'value': trial.value,
                        'state': 'complete'
                    })
                else:
                    all_trials.append({
                        'params': trial.params if hasattr(trial, 'params') else {},
                        'value': None,
                        'state': trial.state.name.lower()
                    })
            
        else:
            # Fallback: Random search with joblib parallel processing
            logger.info("Using random search optimization (Optuna not available)")
            
            # Generate parameter combinations
            param_combinations = []
            for i in range(self.config.n_trials):
                params = param_space.sample_random(random_state=self.config.random_seed + i)
                param_combinations.append(params)
            
            # Parallel evaluation
            if JOBLIB_AVAILABLE and self.config.n_jobs != 1:
                results = Parallel(n_jobs=self.config.n_jobs)(
                    delayed(self.objective_function)(strategy_type, params)
                    for params in param_combinations
                )
            else:
                # Sequential processing
                results = [
                    self.objective_function(strategy_type, params)
                    for params in param_combinations
                ]
            
            # Find best result
            for i, (params, value) in enumerate(zip(param_combinations, results)):
                all_trials.append({
                    'params': params,
                    'value': value,
                    'state': 'complete' if value > -1000 else 'failed'
                })
                
                if value > best_value:
                    best_value = value
                    best_params = params
        
        # Generate best metrics by running the best configuration
        if best_params:
            signals = self._generate_parameterized_signals(strategy_type, best_params)
            best_result = self.backtester._execute_vectorized_backtest(self.data, signals)
            best_metrics = best_result.to_dict()
        
        # Calculate statistics
        completed_trials = len([t for t in all_trials if t['state'] == 'complete'])
        failed_trials = len([t for t in all_trials if t['state'] == 'failed'])
        
        optimization_time = time.time() - start_time
        trials_per_second = len(all_trials) / optimization_time if optimization_time > 0 else 0
        
        result = OptimizationResult(
            strategy_type=strategy_type,
            best_params=best_params,
            best_value=best_value,
            best_metrics=best_metrics,
            n_trials=self.config.n_trials,
            completed_trials=completed_trials,
            failed_trials=failed_trials,
            optimization_time=optimization_time,
            trials_per_second=trials_per_second,
            all_trials=all_trials
        )
        
        logger.info(f"Completed {strategy_type.value}: {completed_trials}/{self.config.n_trials} trials, "
                   f"best {self.config.objective_metric}: {best_value:.3f}")
        
        return result
    
    def optimize_all_strategies(self) -> Dict[StrategyType, OptimizationResult]:
        """Optimize parameters for all available strategies"""
        logger.info(f"Starting optimization for {len(self.parameter_spaces)} strategies...")
        start_time = time.time()
        
        results = {}
        
        for strategy_type in self.parameter_spaces.keys():
            try:
                result = self.optimize_strategy(strategy_type)
                results[strategy_type] = result
            except Exception as e:
                logger.error(f"Optimization failed for {strategy_type.value}: {e}")
                continue
        
        total_time = time.time() - start_time
        logger.info(f"Completed optimization for {len(results)} strategies in {total_time:.2f}s")
        
        return results


def print_optimization_summary(results: Dict[StrategyType, OptimizationResult]) -> None:
    """Print formatted optimization summary"""
    print("\n" + "="*100)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("="*100)
    
    # Sort by best value
    sorted_results = sorted(results.items(), key=lambda x: x[1].best_value, reverse=True)
    
    print(f"{'Strategy':<20} {'Best Value':<12} {'Trials':<8} {'Success%':<9} {'Time(s)':<8} {'TPS':<8}")
    print("-" * 100)
    
    for strategy_type, result in sorted_results:
        success_rate = (result.completed_trials / result.n_trials * 100) if result.n_trials > 0 else 0
        
        print(f"{strategy_type.value:<20} {result.best_value:>10.3f} "
              f"{result.completed_trials:>3d}/{result.n_trials:<3d} {success_rate:>7.1f}% "
              f"{result.optimization_time:>7.1f} {result.trials_per_second:>7.1f}")
    
    print("-" * 100)
    
    # Summary statistics
    total_trials = sum(r.n_trials for r in results.values())
    total_completed = sum(r.completed_trials for r in results.values())
    total_time = sum(r.optimization_time for r in results.values())
    
    print(f"Summary: {len(results)} strategies | {total_completed}/{total_trials} trials completed | "
          f"Total time: {total_time:.1f}s | Avg TPS: {total_completed/total_time:.1f}")
    
    # Best strategies
    if sorted_results:
        print(f"\nTop 3 strategies by {sorted_results[0][1].best_value}:")
        for i, (strategy_type, result) in enumerate(sorted_results[:3]):
            print(f"  {i+1}. {strategy_type.value}: {result.best_value:.3f}")
            print(f"     Best params: {result.best_params}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimization")
    parser.add_argument('--csv', required=True, help='Path to CSV data file')
    parser.add_argument('--strats', default='all', help='Strategies to optimize (all, vol_breakout, etc.)')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--objective', default='sharpe_ratio', 
                       choices=['sharpe_ratio', 'calmar_ratio', 'total_return', 'sortino_ratio'],
                       help='Optimization objective')
    parser.add_argument('--jobs', type=int, default=-1, help='Number of parallel jobs (-1 = all cores)')
    parser.add_argument('--timeout', type=float, help='Optimization timeout in seconds')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load data
        backtester = VectorizedBacktester()
        data = backtester.load_csv_data(args.csv)
        
        # Configure optimization
        config = OptimizationConfig(
            n_trials=args.trials,
            n_jobs=args.jobs,
            timeout=args.timeout,
            objective_metric=args.objective
        )
        
        # Initialize optimizer
        optimizer = StrategyOptimizer(config, data)
        
        # Run optimization
        if args.strats == 'all':
            results = optimizer.optimize_all_strategies()
        else:
            # Single strategy
            try:
                strategy = StrategyType(args.strats)
                result = optimizer.optimize_strategy(strategy)
                results = {strategy: result}
            except ValueError:
                print(f"Unknown strategy: {args.strats}")
                print(f"Available: {[s.value for s in StrategyType]}")
                return 1
        
        # Display results
        print_optimization_summary(results)
        
        # Save results
        if args.output:
            output_data = {
                'config': {
                    'csv_file': args.csv,
                    'strategies': args.strats,
                    'n_trials': args.trials,
                    'objective': args.objective,
                    'n_jobs': args.jobs,
                    'timestamp': datetime.now().isoformat()
                },
                'results': {strategy.value: result.to_dict() for strategy, result in results.items()}
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to {args.output}")
        
        # Performance summary
        total_trials = sum(r.n_trials for r in results.values())
        total_time = sum(r.optimization_time for r in results.values())
        
        print(f"\n⚡ Optimization completed: {total_trials} trials in {total_time:.1f}s")
        print(f"📊 Average throughput: {total_trials/total_time:.1f} trials/second")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())