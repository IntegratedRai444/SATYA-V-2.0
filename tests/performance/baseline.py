"""
Performance Baseline Utility

This module provides functionality to establish and track performance baselines
for the application. It helps in detecting performance regressions by comparing
current performance metrics against historical data.
"""
import json
import statistics
from pathlib import Path
from datetime import datetime
import subprocess
import os
from typing import Dict, List, Optional, Tuple, Union, Any

class PerformanceBaseline:
    """Class to manage performance baselines and detect regressions."""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        """Initialize with the path to the baseline JSON file."""
        self.baseline_file = Path("tests/performance") / baseline_file
        self.baseline_data = self._load_baseline()
        
    def _load_baseline(self) -> Dict:
        """Load existing baseline data or return default structure."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {self.baseline_file}. Starting with empty baseline.")
                    return {}
        return {}
    
    def save_baseline(self) -> None:
        """Save the current baseline data to file."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baseline_data, f, indent=2)
    
    def run_performance_test(
        self, 
        test_name: str, 
        locustfile: str = "test_load.py", 
        users: int = 100, 
        spawn_rate: int = 10, 
        duration: str = "1m"
    ) -> Dict[str, Any]:
        """
        Run a performance test and record the results.
        
        Args:
            test_name: Name of the test scenario
            locustfile: Path to the Locust test file
            users: Number of concurrent users
            spawn_rate: Users spawned per second
            duration: Test duration (e.g., '1m', '5m')
            
        Returns:
            Dictionary containing test results
        """
        print(f"Running performance test: {test_name} with {users} users")
        
        # Run locust and capture output
        cmd = [
            "locust",
            "-f", f"tests/performance/{locustfile}",
            "--headless",
            "--only-summary",
            "--csv=perf_results",
            f"--users={users}",
            f"--spawn-rate={spawn_rate}",
            f"--run-time={duration}",
            "--host=http://localhost:8000"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                check=True
            )
            print(f"Locust output: {result.stdout[:500]}...")  # Print first 500 chars
        except subprocess.CalledProcessError as e:
            print(f"Error running locust: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise
        
        # Parse results (simplified - adjust based on your needs)
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_name": test_name,
            "users": users,
            "spawn_rate": spawn_rate,
            "duration": duration,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "p95_response_time": 0.0,
            "rps": 0.0,
            "failure_ratio": 0.0,
            "total_requests": 0
        }
        
        # Try to parse the CSV output
        try:
            with open("perf_results_stats.csv") as f:
                # Skip header
                next(f)
                # Get the last line (summary)
                for line in f:
                    if line.strip():
                        last_line = line
                
                # Parse the CSV line
                parts = last_line.split(',')
                if len(parts) >= 10:  # Ensure we have enough columns
                    stats.update({
                        "total_requests": int(parts[2]),
                        "failure_ratio": float(parts[3]),
                        "success_rate": 1.0 - float(parts[3]),
                        "avg_response_time": float(parts[5]),
                        "p95_response_time": float(parts[8]),
                        "rps": float(parts[9])
                    })
        except Exception as e:
            print(f"Warning: Could not parse performance results: {e}")
        
        # Add to baseline
        if test_name not in self.baseline_data:
            self.baseline_data[test_name] = []
        
        self.baseline_data[test_name].append(stats)
        self.save_baseline()
        
        return stats
    
    def check_performance_regression(
        self, 
        test_name: str, 
        threshold_pct: float = 10.0,
        min_runs: int = 2
    ) -> Tuple[bool, str]:
        """
        Check if performance has regressed beyond the threshold.
        
        Args:
            test_name: Name of the test to check
            threshold_pct: Percentage threshold for considering a regression
            min_runs: Minimum number of runs required for comparison
            
        Returns:
            Tuple of (has_regression, message)
        """
        if test_name not in self.baseline_data:
            return False, f"No baseline data for test: {test_name}"
            
        history = self.baseline_data[test_name]
        if len(history) < min_runs:
            return False, f"Not enough runs for comparison (need at least {min_runs})"
            
        current = history[-1]
        previous = history[-2] if len(history) > 1 else None
        
        if not previous:
            return False, "No previous run to compare with"
            
        regressions = []
        
        # Check response time
        if 'avg_response_time' in current and 'avg_response_time' in previous:
            if previous['avg_response_time'] > 0:  # Avoid division by zero
                increase = ((current['avg_response_time'] - previous['avg_response_time']) / 
                          previous['avg_response_time']) * 100
                if increase > threshold_pct:
                    regressions.append(
                        f"Average response time increased by {increase:.2f}% "
                        f"(from {previous['avg_response_time']:.2f}ms to {current['avg_response_time']:.2f}ms)"
                    )
        
        # Check success rate
        if 'success_rate' in current and 'success_rate' in previous:
            if previous['success_rate'] > 0:  # Avoid division by zero
                decrease = ((previous['success_rate'] - current['success_rate']) / 
                          previous['success_rate']) * 100
                if decrease > threshold_pct:
                    regressions.append(
                        f"Success rate decreased by {decrease:.2f}% "
                        f"(from {previous['success_rate']*100:.1f}% to {current['success_rate']*100:.1f}%)"
                    )
        
        # Check RPS (higher is better)
        if 'rps' in current and 'rps' in previous and previous['rps'] > 0:
            decrease = ((previous['rps'] - current['rps']) / previous['rps']) * 100
            if decrease > threshold_pct:
                regressions.append(
                    f"Requests per second decreased by {decrease:.2f}% "
                    f"(from {previous['rps']:.1f} to {current['rps']:.1f})"
                )
        
        if regressions:
            return True, " | ".join(regressions)
        return False, "No significant regression detected"

def main():
    """Run performance tests and check for regressions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run performance tests and check for regressions.')
    parser.add_argument('--test', type=str, help='Name of the test to run')
    parser.add_argument('--users', type=int, default=100, help='Number of users')
    parser.add_argument('--spawn-rate', type=int, default=10, help='Users spawned per second')
    parser.add_argument('--duration', type=str, default='1m', help='Test duration (e.g., 1m, 5m)')
    parser.add_argument('--threshold', type=float, default=10.0, 
                       help='Percentage threshold for regression detection')
    
    args = parser.parse_args()
    
    baseline = PerformanceBaseline()
    
    # Define test scenarios
    test_scenarios = [
        {"name": "normal_load", "users": 100, "spawn_rate": 10, "duration": "1m"},
        {"name": "stress_test", "users": 500, "spawn_rate": 50, "duration": "2m"},
    ]
    
    # If a specific test is provided, run only that one
    if args.test:
        test_scenarios = [t for t in test_scenarios if t["name"] == args.test]
        if not test_scenarios:
            test_scenarios = [{
                "name": args.test, 
                "users": args.users, 
                "spawn_rate": args.spawn_rate, 
                "duration": args.duration
            }]
    
    # Run tests
    for scenario in test_scenarios:
        print(f"\n=== Running test: {scenario['name']} ===")
        
        # Run the performance test
        baseline.run_performance_test(
            test_name=scenario["name"],
            users=scenario["users"],
            spawn_rate=scenario["spawn_rate"],
            duration=scenario["duration"]
        )
        
        # Check for regressions
        has_regression, message = baseline.check_performance_regression(
            test_name=scenario["name"],
            threshold_pct=args.threshold
        )
        
        if has_regression:
            print(f"\n❌ PERFORMANCE REGRESSION DETECTED in {scenario['name']}:")
            print(f"   {message}")
            # Exit with error code if regression is detected
            exit(1)
        else:
            print(f"\n✅ Performance test passed for {scenario['name']}")
            print(f"   {message}")

if __name__ == "__main__":
    main()
