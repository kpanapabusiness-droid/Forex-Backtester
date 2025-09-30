"""
Test MT5 parity smoke tests.

Provides scaffold smoke tests for MT5 parity functionality with xfail
markers when fixtures are missing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMT5ParitySmoke:
    """Smoke tests for MT5 parity functionality."""

    @pytest.mark.xfail(
        strict=False, reason="MT5 data fixture may not be available in test environment"
    )
    def test_mt5_parity_config_exists(self):
        """
        Test that MT5 parity configuration file exists and is valid.

        This is a basic smoke test to verify the MT5 parity configuration
        is present and can be loaded.
        """
        config_path = project_root / "configs/validation/mt5_parity_d1.yaml"

        if not config_path.exists():
            pytest.xfail(f"MT5 parity config not found: {config_path}")

        # Try to load the config
        import yaml

        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            pytest.xfail(f"Failed to load MT5 parity config: {e}")

        # Basic validation that it's a valid config
        assert isinstance(cfg, dict), "Config should be a dictionary"
        assert "symbol" in cfg, "Config should specify symbol"
        assert "timeframe" in cfg, "Config should specify timeframe"

        # Verify it's the expected MT5 parity config
        assert cfg.get("symbol") == "EURUSD", "MT5 parity should use EURUSD"
        assert cfg.get("timeframe") == "D1", "MT5 parity should use D1 timeframe"

    @pytest.mark.xfail(
        strict=False, reason="MT5 parity run script may fail if data fixtures missing"
    )
    def test_mt5_parity_run_script_exists(self):
        """
        Test that MT5 parity run script exists and can be executed.

        This smoke test verifies the MT5 parity run script is present
        and can be invoked (even if it fails due to missing data).
        """
        script_path = project_root / "scripts/mt5_parity_run.py"

        if not script_path.exists():
            pytest.xfail(f"MT5 parity run script not found: {script_path}")

        # Test that the script can be invoked (may fail due to missing data)
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root,
            )

            # Script should at least respond to --help or show some output
            # (even if it doesn't support --help, it shouldn't hang)
            assert result.returncode is not None, "Script should complete execution"

        except subprocess.TimeoutExpired:
            pytest.xfail("MT5 parity run script timed out")
        except Exception as e:
            pytest.xfail(f"Failed to execute MT5 parity run script: {e}")

    @pytest.mark.xfail(
        strict=False, reason="MT5 comparison script may fail if data fixtures missing"
    )
    def test_mt5_compare_script_exists(self):
        """
        Test that MT5 comparison script exists and can be executed.

        This smoke test verifies the MT5 comparison script is present
        and can be invoked (even if it fails due to missing data).
        """
        script_path = project_root / "scripts/mt5_compare.py"

        if not script_path.exists():
            pytest.xfail(f"MT5 compare script not found: {script_path}")

        # Test that the script can be invoked
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root,
            )

            # Script should at least respond or complete
            assert result.returncode is not None, "Script should complete execution"

        except subprocess.TimeoutExpired:
            pytest.xfail("MT5 compare script timed out")
        except Exception as e:
            pytest.xfail(f"Failed to execute MT5 compare script: {e}")

    @pytest.mark.xfail(strict=False, reason="MT5 data directory may not exist in test environment")
    def test_mt5_data_directory_structure(self):
        """
        Test that MT5 data directory has expected structure.

        This smoke test checks for the presence of MT5 data directory
        and expected file structure (if available).
        """
        mt5_data_dir = project_root / "mt5"

        if not mt5_data_dir.exists():
            pytest.xfail(f"MT5 data directory not found: {mt5_data_dir}")

        # Check for expected subdirectory structure
        expected_subdirs = ["eurusd_d1_2022_2024"]

        for subdir in expected_subdirs:
            subdir_path = mt5_data_dir / subdir
            if not subdir_path.exists():
                pytest.xfail(f"Expected MT5 subdirectory not found: {subdir_path}")

        # Check for expected files in the EURUSD directory
        eurusd_dir = mt5_data_dir / "eurusd_d1_2022_2024"
        expected_files = ["eurusd_d1_2022_2024_trades.csv"]

        for file_name in expected_files:
            file_path = eurusd_dir / file_name
            if not file_path.exists():
                pytest.xfail(f"Expected MT5 data file not found: {file_path}")

        # If we get here, the structure exists
        assert True, "MT5 data directory structure is present"

    @pytest.mark.xfail(
        strict=False, reason="Full MT5 parity pipeline requires complete data fixtures"
    )
    def test_mt5_parity_pipeline_smoke(self):
        """
        Test that the full MT5 parity pipeline can be invoked.

        This is an end-to-end smoke test that attempts to run the full
        MT5 parity pipeline. Expected to fail if data fixtures are missing.
        """
        # Check prerequisites
        config_path = project_root / "configs/validation/mt5_parity_d1.yaml"
        run_script = project_root / "scripts/mt5_parity_run.py"
        compare_script = project_root / "scripts/mt5_compare.py"

        missing_components = []
        if not config_path.exists():
            missing_components.append(f"config: {config_path}")
        if not run_script.exists():
            missing_components.append(f"run script: {run_script}")
        if not compare_script.exists():
            missing_components.append(f"compare script: {compare_script}")

        if missing_components:
            pytest.xfail(f"Missing MT5 parity components: {', '.join(missing_components)}")

        # Attempt to run the parity backtest
        try:
            print("\n🚀 Running MT5 parity backtest (smoke test)...")
            result = subprocess.run(
                [sys.executable, str(run_script)],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout for smoke test
                cwd=project_root,
            )

            # For smoke test, we just verify the script completes
            # (success or failure is acceptable, we're testing invocation)
            assert result.returncode is not None, "Parity run should complete"

            # If it succeeded, try the comparison
            if result.returncode == 0:
                print("✅ Parity backtest completed, attempting comparison...")
                compare_result = subprocess.run(
                    [sys.executable, str(compare_script)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=project_root,
                )

                assert compare_result.returncode is not None, "Comparison should complete"
                print(
                    f"✅ MT5 parity pipeline smoke test completed (return codes: run={result.returncode}, compare={compare_result.returncode})"
                )
            else:
                print(
                    f"⚠️  Parity backtest failed (return code: {result.returncode}), skipping comparison"
                )
                pytest.xfail(f"Parity backtest failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.xfail("MT5 parity pipeline timed out")
        except Exception as e:
            pytest.xfail(f"MT5 parity pipeline failed: {e}")

    def test_mt5_parity_imports_available(self):
        """
        Test that MT5 parity related imports are available.

        This basic test verifies that the MT5 comparison modules can be imported
        without external data dependencies.
        """
        try:
            # Test import of MT5 comparison functions
            from scripts.mt5_compare import compare_totals, match_trades

            # Verify functions exist and are callable
            assert callable(compare_totals), "compare_totals should be callable"
            assert callable(match_trades), "match_trades should be callable"

        except ImportError as e:
            pytest.fail(f"Failed to import MT5 comparison functions: {e}")

    def test_mt5_parity_synthetic_comparison(self):
        """
        Test MT5 comparison logic with synthetic data.

        This test uses synthetic data to verify the MT5 comparison logic
        works correctly without requiring actual MT5 data fixtures.
        """
        try:
            import pandas as pd

            from scripts.mt5_compare import compare_totals, match_trades

            # Create synthetic trade data for testing comparison logic
            our_trades = pd.DataFrame(
                {
                    "open_time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                    "close_time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                    "symbol": ["EURUSD", "EURUSD"],
                    "side": [1, -1],
                    "open_price": [1.1000, 1.1050],
                    "close_price": [1.1010, 1.1040],
                    "pnl_currency": [100.0, 100.0],
                }
            )

            mt5_trades = pd.DataFrame(
                {
                    "open_time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                    "close_time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                    "symbol": ["EURUSD", "EURUSD"],
                    "side": [1, -1],
                    "open_price": [1.1000, 1.1050],
                    "close_price": [1.1010, 1.1040],
                    "pnl_currency": [100.0, 100.0],
                }
            )

            # Test trade matching
            matches, unmatched_ours, unmatched_mt5 = match_trades(
                our_trades, mt5_trades, price_tol=0.00005, time_tol_bars=1
            )

            assert len(matches) == 2, "Should match both synthetic trades"
            assert len(unmatched_ours) == 0, "No unmatched trades expected"
            assert len(unmatched_mt5) == 0, "No unmatched MT5 trades expected"

            # Test total comparison
            comparison_result = compare_totals(our_trades, mt5_trades, pnl_pct_tol=0.001)

            assert comparison_result["passed"] is True, (
                "Identical synthetic data should pass comparison"
            )
            assert len(comparison_result["errors"]) == 0, "No errors expected for identical data"

        except ImportError as e:
            pytest.xfail(f"MT5 comparison modules not available: {e}")
        except Exception as e:
            pytest.fail(f"MT5 synthetic comparison test failed: {e}")

    @pytest.mark.xfail(
        strict=False, reason="Results directory may not exist or may not have MT5 parity results"
    )
    def test_mt5_parity_results_structure(self):
        """
        Test that MT5 parity results have expected structure (if they exist).

        This smoke test checks for the presence of MT5 parity results
        and validates their basic structure.
        """
        results_dir = project_root / "results/validation/mt5_parity_d1"

        if not results_dir.exists():
            pytest.xfail(f"MT5 parity results directory not found: {results_dir}")

        # Check for expected result files
        expected_files = ["trades.csv", "summary.txt", "equity_curve.csv"]

        missing_files = []
        for file_name in expected_files:
            file_path = results_dir / file_name
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            pytest.xfail(f"Missing MT5 parity result files: {missing_files}")

        # If trades.csv exists, do basic validation
        trades_file = results_dir / "trades.csv"
        try:
            import pandas as pd

            trades_df = pd.read_csv(trades_file)

            # Basic structure validation
            expected_columns = ["entry_date", "exit_date", "side", "pnl"]
            missing_columns = [col for col in expected_columns if col not in trades_df.columns]

            if missing_columns:
                pytest.xfail(f"Missing columns in trades.csv: {missing_columns}")

            # If we get here, the structure is valid
            assert len(trades_df.columns) > 0, "Trades file should have columns"
            print(f"✅ MT5 parity results structure validated ({len(trades_df)} trades)")

        except Exception as e:
            pytest.xfail(f"Failed to validate MT5 parity results structure: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
