"""
Performance testing script for Phase E CLI tools
Tests 100k-row CSV completion time and validates <2 min requirement
"""

import time
import csv
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

def generate_test_csv(filename: str, n_rows: int = 100000):
    """Generate test market data CSV"""
    print(f"Generating {n_rows:,} row test CSV: {filename}")
    start_time = time.time()
    
    base_price = 50000.0  # Starting price
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Generate data
        current_time = datetime(2023, 1, 1)
        price = base_price
        
        for i in range(n_rows):
            # Random walk price
            change = (hash(str(i) + "price") % 1000 - 500) / 10000  # Deterministic "random"
            price *= (1 + change)
            
            # OHLC generation
            open_price = price
            volatility = abs(change) * 2
            high_price = price * (1 + volatility)
            low_price = price * (1 - volatility)
            close_price = price * (1 + change * 0.5)
            
            volume = 1000 + (hash(str(i) + "vol") % 5000)
            
            writer.writerow([
                current_time.isoformat(),
                f"{open_price:.2f}",
                f"{high_price:.2f}",
                f"{low_price:.2f}",
                f"{close_price:.2f}",
                volume
            ])
            
            current_time += timedelta(minutes=1)
    
    gen_time = time.time() - start_time
    print(f"Generated {n_rows:,} rows in {gen_time:.2f}s")
    return filename

def test_backtest_performance(csv_file: str):
    """Test backtest CLI performance"""
    print(f"\nTesting backtest performance with {csv_file}")
    
    # Test command
    backtest_script = "src/project_chimera/cli/backtest.py"
    
    if not os.path.exists(backtest_script):
        print(f"❌ Backtest script not found: {backtest_script}")
        return False
    
    cmd = [
        sys.executable, backtest_script,
        '--csv', csv_file,
        '--strats', 'vol_breakout'  # Test single strategy first
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Performance target (<120s): {'✅ PASS' if execution_time < 120 else '❌ FAIL'}")
        
        if result.returncode == 0:
            print("✅ Backtest completed successfully")
            if "rows/second" in result.stdout:
                print(f"Output: {result.stdout}")
            return execution_time < 120
        else:
            print(f"❌ Backtest failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Backtest timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return False

def test_optimize_performance(csv_file: str):
    """Test optimization CLI performance"""
    print(f"\nTesting optimization performance with {csv_file}")
    
    optimize_script = "src/project_chimera/cli/optimise.py"
    
    if not os.path.exists(optimize_script):
        print(f"❌ Optimize script not found: {optimize_script}")
        return False
    
    cmd = [
        sys.executable, optimize_script,
        '--csv', csv_file,
        '--strats', 'vol_breakout',
        '--trials', '10',  # Reduced for testing
        '--jobs', '2'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"Optimization time: {execution_time:.2f}s")
        
        if result.returncode == 0:
            print("✅ Optimization completed successfully")
            if "trials/second" in result.stdout:
                print(f"Output: {result.stdout}")
            return True
        else:
            print(f"❌ Optimization failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Optimization timed out")
        return False
    except Exception as e:
        print(f"❌ Error running optimization: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with smaller dataset"""
    print("\nTesting basic functionality with 1k rows...")
    
    # Generate small test file
    small_csv = "test_data_1k.csv"
    generate_test_csv(small_csv, 1000)
    
    try:
        # Test backtest
        backtest_script = "src/project_chimera/cli/backtest.py"
        cmd = [sys.executable, backtest_script, '--csv', small_csv, '--strats', 'mini_momo']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Basic backtest functionality works")
        else:
            print(f"❌ Basic backtest failed: {result.stderr}")
            return False
        
        # Test optimize
        optimize_script = "src/project_chimera/cli/optimise.py"
        cmd = [sys.executable, optimize_script, '--csv', small_csv, '--strats', 'mini_momo', '--trials', '3']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Basic optimization functionality works")
        else:
            print(f"❌ Basic optimization failed: {result.stderr}")
            return False
        
        return True
        
    finally:
        # Cleanup
        if os.path.exists(small_csv):
            os.remove(small_csv)

def main():
    """Main testing function"""
    print("="*80)
    print("PHASE E PERFORMANCE TESTING")
    print("="*80)
    
    # Test basic functionality first
    if not test_basic_functionality():
        print("❌ Basic functionality tests failed")
        return 1
    
    # Generate 100k row test file
    test_csv = "test_data_100k.csv"
    
    try:
        generate_test_csv(test_csv, 100000)
        
        # Test performance
        backtest_pass = test_backtest_performance(test_csv)
        optimize_pass = test_optimize_performance(test_csv)
        
        print("\n" + "="*80)
        print("PERFORMANCE TEST RESULTS")
        print("="*80)
        print(f"100k-row backtest <2min: {'✅ PASS' if backtest_pass else '❌ FAIL'}")
        print(f"Optimization functional: {'✅ PASS' if optimize_pass else '❌ FAIL'}")
        
        overall_pass = backtest_pass and optimize_pass
        print(f"Overall Phase E: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        
        return 0 if overall_pass else 1
        
    finally:
        # Cleanup
        if os.path.exists(test_csv):
            os.remove(test_csv)
            print(f"Cleaned up {test_csv}")

if __name__ == '__main__':
    exit(main())