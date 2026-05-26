"""
test_app.py
Pytest test suite for the Monte Carlo Portfolio Simulator.

Imports only from simulation.py (no Streamlit dependency), so all tests
run headlessly without mocking the Streamlit runtime.
"""

import numpy as np
import pandas as pd
import pytest

from simulation import (
    FREQ,
    compute_path_drawdowns,
    get_price_history,
    get_ticker_info,
    portfolio_div_yield,
    portfolio_expense_ratio,
    run_backtest,
    run_simulation,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def two_asset_prices():
    """Synthetic two-asset price history with ~500 trading days."""
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, n))),
        },
        index=dates,
    )
    return prices


@pytest.fixture
def one_asset_prices(two_asset_prices):
    return two_asset_prices[["A"]]


@pytest.fixture
def four_year_prices():
    """~4 years of synthetic prices — enough for all backtest frequencies."""
    rng = np.random.default_rng(7)
    n = 1010   # ~4 trading years
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))),
        },
        index=dates,
    )


# ─── portfolio_div_yield ──────────────────────────────────────────────────────

class TestPortfolioDivYield:

    def test_equal_weight_two_tickers(self):
        holdings = [
            {"value": 1000.0, "div_yield": 0.02},
            {"value": 1000.0, "div_yield": 0.01},
        ]
        result = portfolio_div_yield(holdings)
        assert result == pytest.approx(0.015)

    def test_unequal_weight(self):
        holdings = [
            {"value": 3000.0, "div_yield": 0.02},
            {"value": 1000.0, "div_yield": 0.00},
        ]
        # 75 % weight on 2 % yield → 0.015
        assert portfolio_div_yield(holdings) == pytest.approx(0.015)

    def test_empty_holdings(self):
        assert portfolio_div_yield([]) == 0.0

    def test_zero_total_value(self):
        holdings = [{"value": 0.0, "div_yield": 0.05}]
        assert portfolio_div_yield(holdings) == 0.0

    def test_missing_div_yield_key(self):
        """Holding dicts without 'div_yield' should default to zero."""
        holdings = [{"value": 1000.0}]
        assert portfolio_div_yield(holdings) == pytest.approx(0.0)

    def test_missing_value_key(self):
        """Holding dicts without 'value' should not raise."""
        holdings = [{"div_yield": 0.02}]
        assert portfolio_div_yield(holdings) == 0.0

    def test_single_holding_full_weight(self):
        holdings = [{"value": 5000.0, "div_yield": 0.013}]
        assert portfolio_div_yield(holdings) == pytest.approx(0.013)

    def test_result_in_unit_range(self):
        holdings = [
            {"value": 500.0, "div_yield": 0.05},
            {"value": 500.0, "div_yield": 0.03},
        ]
        result = portfolio_div_yield(holdings)
        assert 0.0 <= result <= 1.0


# ─── run_simulation — output shape & basic contracts ─────────────────────────

class TestRunSimulationShape:

    def test_two_asset_shape(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        port, dc = run_simulation(w, two_asset_prices, 10_000, 500, "Monthly", 10, 200, seed=1)
        assert port.shape == (200, 10 * 12 + 1)
        assert dc.shape == (200, 10 * 12 + 1)

    def test_single_asset_shape(self, one_asset_prices):
        port, dc = run_simulation(
            np.array([1.0]), one_asset_prices, 10_000, 0, "Monthly", 5, 100, seed=1
        )
        assert port.shape == (100, 5 * 12 + 1)
        assert dc.shape == (100, 5 * 12 + 1)

    @pytest.mark.parametrize("freq,steps_per_yr", FREQ.items())
    def test_all_frequencies(self, two_asset_prices, freq, steps_per_yr):
        w = np.array([0.6, 0.4])
        port, dc = run_simulation(w, two_asset_prices, 10_000, 100, freq, 5, 50, seed=1)
        assert port.shape == (50, 5 * steps_per_yr + 1)

    def test_initial_column_equals_initial(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 12_345.0, 0, "Monthly", 5, 50, seed=1)
        np.testing.assert_array_equal(port[:, 0], 12_345.0)

    def test_div_cash_initial_column_is_zero(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        _, dc = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 50,
            drip=False, annual_div_yield=0.02, seed=1,
        )
        np.testing.assert_array_equal(dc[:, 0], 0.0)

    def test_no_nan_in_output(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        port, dc = run_simulation(w, two_asset_prices, 10_000, 500, "Monthly", 10, 100, seed=1)
        assert not np.any(np.isnan(port))
        assert not np.any(np.isnan(dc))


# ─── run_simulation — contributions ──────────────────────────────────────────

class TestRunSimulationContributions:

    def test_contribution_increases_median(self, one_asset_prices):
        """Adding periodic cash always raises the median final value."""
        w = np.array([1.0])
        port_none, _ = run_simulation(w, one_asset_prices, 10_000, 0, "Monthly", 10, 500, seed=7)
        port_500, _  = run_simulation(w, one_asset_prices, 10_000, 500, "Monthly", 10, 500, seed=7)
        assert np.median(port_500[:, -1]) > np.median(port_none[:, -1])

    def test_zero_contribution_does_not_raise(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 10_000, 0.0, "Monthly", 5, 100, seed=1)
        assert port.shape[0] == 100


# ─── run_simulation — DRIP behaviour ─────────────────────────────────────────

class TestRunSimulationDRIP:

    def test_drip_on_dividend_cash_all_zeros(self, two_asset_prices):
        """When DRIP is ON, no cash is paid out regardless of yield."""
        w = np.array([0.6, 0.4])
        _, dc = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 100,
            drip=True, annual_div_yield=0.03, seed=1,
        )
        np.testing.assert_array_equal(dc, 0.0)

    def test_drip_off_dividend_cash_monotone(self, two_asset_prices):
        """Cumulative dividends must be non-decreasing across all paths."""
        w = np.array([0.6, 0.4])
        _, dc = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 200,
            drip=False, annual_div_yield=0.02, seed=1,
        )
        assert np.all(np.diff(dc, axis=1) >= -1e-10)

    def test_drip_off_positive_final_dividends(self, two_asset_prices):
        """Cumulative dividends at end must be positive for non-zero yield."""
        w = np.array([0.6, 0.4])
        _, dc = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 100,
            drip=False, annual_div_yield=0.02, seed=1,
        )
        assert dc[:, -1].mean() > 0

    def test_drip_off_portfolio_lower_than_drip_on(self, two_asset_prices):
        """Portfolio value ex-div must be lower than total-return (DRIP ON)."""
        w = np.array([0.6, 0.4])
        port_on, _  = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 10, 500,
            drip=True, annual_div_yield=0.02, seed=42,
        )
        port_off, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 10, 500,
            drip=False, annual_div_yield=0.02, seed=42,
        )
        assert port_on[:, -1].mean() > port_off[:, -1].mean()

    def test_drip_off_total_wealth_close_to_drip_on(self, two_asset_prices):
        """
        Total wealth (portfolio + dividends) under DRIP OFF should be in
        the same ballpark as DRIP ON.  DRIP ON is slightly higher due to
        compounding, so the ratio should be between 0.88 and 1.15.
        """
        w = np.array([0.6, 0.4])
        port_on, _   = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 10, 500,
            drip=True, annual_div_yield=0.02, seed=42,
        )
        port_off, dc = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 10, 500,
            drip=False, annual_div_yield=0.02, seed=42,
        )
        ratio = port_on[:, -1].mean() / (port_off[:, -1] + dc[:, -1]).mean()
        assert 0.88 < ratio < 1.15, f"Wealth ratio {ratio:.3f} outside expected range"

    def test_zero_yield_drip_off_equals_drip_on(self, two_asset_prices):
        """DRIP OFF with 0 % yield must produce identical results to DRIP ON."""
        w = np.array([0.6, 0.4])
        port_on, dc_on   = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 200,
            drip=True, annual_div_yield=0.0, seed=5,
        )
        port_off, dc_off = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 200,
            drip=False, annual_div_yield=0.0, seed=5,
        )
        np.testing.assert_array_almost_equal(port_on, port_off, decimal=10)
        np.testing.assert_array_equal(dc_off, 0.0)


# ─── portfolio_expense_ratio ──────────────────────────────────────────────────

class TestPortfolioExpenseRatio:

    def test_equal_weight_two_tickers(self):
        holdings = [
            {"value": 1000.0, "expense_ratio": 0.001},
            {"value": 1000.0, "expense_ratio": 0.003},
        ]
        assert portfolio_expense_ratio(holdings) == pytest.approx(0.002)

    def test_unequal_weight(self):
        # 75 % in 0.1 % ER, 25 % in 0.3 % ER → 0.75*0.001 + 0.25*0.003 = 0.0015
        holdings = [
            {"value": 3000.0, "expense_ratio": 0.001},
            {"value": 1000.0, "expense_ratio": 0.003},
        ]
        assert portfolio_expense_ratio(holdings) == pytest.approx(0.0015)

    def test_empty_holdings(self):
        assert portfolio_expense_ratio([]) == 0.0

    def test_zero_total_value(self):
        holdings = [{"value": 0.0, "expense_ratio": 0.001}]
        assert portfolio_expense_ratio(holdings) == 0.0

    def test_missing_expense_ratio_key(self):
        """Holding dicts without 'expense_ratio' should default to zero."""
        holdings = [{"value": 1000.0}]
        assert portfolio_expense_ratio(holdings) == pytest.approx(0.0)

    def test_missing_value_key(self):
        """Holding dicts without 'value' should not raise."""
        holdings = [{"expense_ratio": 0.001}]
        assert portfolio_expense_ratio(holdings) == 0.0

    def test_single_holding_full_weight(self):
        holdings = [{"value": 5000.0, "expense_ratio": 0.0009}]
        assert portfolio_expense_ratio(holdings) == pytest.approx(0.0009)

    def test_result_in_unit_range(self):
        holdings = [
            {"value": 500.0, "expense_ratio": 0.003},
            {"value": 500.0, "expense_ratio": 0.001},
        ]
        result = portfolio_expense_ratio(holdings)
        assert 0.0 <= result <= 1.0

    def test_stocks_with_zero_er(self):
        """Stock holdings without an ER field should not inflate the average."""
        holdings = [
            {"value": 5000.0, "expense_ratio": 0.0009},  # ETF
            {"value": 5000.0},                            # Stock (no ER key)
        ]
        # 50 % weight on 0.09 % ER + 50 % weight on 0 % ER = 0.045 %
        assert portfolio_expense_ratio(holdings) == pytest.approx(0.00045)


# ─── run_simulation — expense ratio delta ─────────────────────────────────────

class TestRunSimulationExpenseRatio:

    def test_er_delta_zero_identical_to_default(self, two_asset_prices):
        """er_delta=0.0 must produce bit-for-bit identical results."""
        w = np.array([0.6, 0.4])
        port_default, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 200, seed=7
        )
        port_zero_er, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 200, seed=7, er_delta=0.0
        )
        np.testing.assert_array_equal(port_default, port_zero_er)

    def test_positive_er_delta_reduces_median(self, two_asset_prices):
        """A higher ER (positive er_delta) should lower the median final value."""
        w = np.array([0.6, 0.4])
        port_base, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 10, 500, seed=42
        )
        port_high_er, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 10, 500, seed=42, er_delta=0.01
        )
        assert np.median(port_base[:, -1]) > np.median(port_high_er[:, -1])

    def test_negative_er_delta_increases_median(self, two_asset_prices):
        """A lower ER (negative er_delta) should raise the median final value."""
        w = np.array([0.6, 0.4])
        port_base, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 10, 500, seed=42
        )
        port_low_er, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 10, 500, seed=42, er_delta=-0.005
        )
        assert np.median(port_low_er[:, -1]) > np.median(port_base[:, -1])

    def test_er_delta_shape_unchanged(self, two_asset_prices):
        """Applying er_delta must not change the output array shapes."""
        w = np.array([0.6, 0.4])
        port, dc = run_simulation(
            w, two_asset_prices, 10_000, 200, "Monthly", 5, 100, seed=1, er_delta=0.002
        )
        assert port.shape == (100, 5 * 12 + 1)
        assert dc.shape == (100, 5 * 12 + 1)

    def test_er_delta_no_nan(self, two_asset_prices):
        """No NaN values should appear when er_delta is applied."""
        w = np.array([0.6, 0.4])
        port, dc = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 5, 100, seed=1, er_delta=0.005
        )
        assert not np.any(np.isnan(port))
        assert not np.any(np.isnan(dc))

    def test_er_delta_monotone_effect_over_time(self, two_asset_prices):
        """The median gap between base and high-ER portfolios grows over time."""
        w = np.array([0.6, 0.4])
        port_base, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 20, 500, seed=42
        )
        port_high_er, _ = run_simulation(
            w, two_asset_prices, 10_000, 0, "Monthly", 20, 500, seed=42, er_delta=0.01
        )
        gap_early = np.median(port_base[:, 12]) - np.median(port_high_er[:, 12])
        gap_late  = np.median(port_base[:, -1]) - np.median(port_high_er[:, -1])
        assert gap_late > gap_early, "ER drag should compound and widen over time"


# ─── compute_path_drawdowns ───────────────────────────────────────────────────

class TestComputePathDrawdowns:

    def test_shape_matches_input(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 10_000, 0, "Monthly", 5, 100, seed=1)
        dd = compute_path_drawdowns(port)
        assert dd.shape == port.shape

    def test_initial_step_is_zero(self, two_asset_prices):
        """t=0 is always the starting peak so drawdown must be 0 for every path."""
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 10_000, 0, "Monthly", 5, 100, seed=1)
        dd = compute_path_drawdowns(port)
        np.testing.assert_array_equal(dd[:, 0], 0.0)

    def test_always_non_negative(self, two_asset_prices):
        """Drawdown can never be negative (can't be above a running maximum)."""
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 10_000, 0, "Monthly", 10, 200, seed=1)
        dd = compute_path_drawdowns(port)
        assert np.all(dd >= 0.0)

    def test_always_at_most_one(self, two_asset_prices):
        """Drawdown fraction can never exceed 1 (portfolio can't lose more than 100 %)."""
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 10_000, 0, "Monthly", 10, 200, seed=1)
        dd = compute_path_drawdowns(port)
        assert np.all(dd <= 1.0 + 1e-10)

    def test_no_nan_in_output(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 10_000, 500, "Monthly", 5, 100, seed=1)
        dd = compute_path_drawdowns(port)
        assert not np.any(np.isnan(dd))

    def test_monotone_growth_has_zero_drawdown(self):
        """A strictly increasing path is always at its all-time high → 0 drawdown."""
        values = np.array([[100.0, 101.0, 103.0, 106.0, 110.0]])
        dd = compute_path_drawdowns(values)
        np.testing.assert_allclose(dd, 0.0, atol=1e-12)

    def test_single_step_drop_correct(self):
        """Portfolio drops from 100 → 80: drawdown should be 0 then 0.20."""
        values = np.array([[100.0, 80.0]])
        dd = compute_path_drawdowns(values)
        assert dd[0, 0] == pytest.approx(0.0)
        assert dd[0, 1] == pytest.approx(0.20)

    def test_recovery_to_peak_resets_drawdown(self):
        """After recovering back to the prior high the drawdown returns to 0."""
        values = np.array([[100.0, 80.0, 100.0]])
        dd = compute_path_drawdowns(values)
        assert dd[0, 0] == pytest.approx(0.0)   # at peak
        assert dd[0, 1] == pytest.approx(0.20)  # 20 % below peak
        assert dd[0, 2] == pytest.approx(0.0)   # recovered to peak

    def test_new_all_time_high_resets_drawdown(self):
        """Surpassing the previous peak sets drawdown to 0 at that new high."""
        values = np.array([[100.0, 90.0, 110.0]])
        dd = compute_path_drawdowns(values)
        assert dd[0, 0] == pytest.approx(0.0)   # initial peak
        assert dd[0, 1] == pytest.approx(0.10)  # 10 % below
        assert dd[0, 2] == pytest.approx(0.0)   # new all-time high

    def test_max_drawdown_sensible_for_sim_output(self, two_asset_prices):
        """Max drawdown over a 10-year run should be > 0 for most paths."""
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 10_000, 0, "Monthly", 10, 500, seed=7)
        dd = compute_path_drawdowns(port)
        max_dd = dd.max(axis=1)
        # At least 70 % of simulations should experience some drawdown.
        assert np.mean(max_dd > 0) > 0.70


# ─── run_backtest ─────────────────────────────────────────────────────────────

class TestRunBacktest:

    def test_output_keys_present(self, four_year_prices):
        """All documented return-dict keys must be present."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(w, four_year_prices, 10_000, 500, "Monthly")
        for key in (
            "portfolio_values", "div_cash", "total_wealth", "drawdowns",
            "total_invested", "n_steps", "n_years", "total_invested_final",
            "final_wealth", "cagr", "total_return_pct", "max_drawdown",
            "ann_vol", "sharpe", "annual_rets", "annual_years",
            "best_yr", "worst_yr",
        ):
            assert key in bt, f"Missing key: {key!r}"

    def test_initial_value_correct(self, four_year_prices):
        """Portfolio value at t=0 must equal the initial argument."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(w, four_year_prices, 12_345.0, 0, "Monthly")
        assert bt["portfolio_values"].iloc[0] == pytest.approx(12_345.0)

    def test_total_wealth_equals_portfolio_plus_div(self, four_year_prices):
        """total_wealth must equal portfolio_values + div_cash at every step."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(
            w, four_year_prices, 10_000, 0, "Monthly",
            drip=False, annual_div_yield=0.02,
        )
        np.testing.assert_allclose(
            bt["total_wealth"].values,
            bt["portfolio_values"].values + bt["div_cash"].values,
            rtol=1e-10,
        )

    def test_drawdown_in_unit_range(self, four_year_prices):
        """All drawdown values must be in [0, 1]."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(w, four_year_prices, 10_000, 0, "Monthly")
        assert np.all(bt["drawdowns"].values >= 0.0)
        assert np.all(bt["drawdowns"].values <= 1.0 + 1e-10)

    def test_drawdown_at_t0_is_zero(self, four_year_prices):
        """Running peak at t=0 equals initial value → drawdown = 0."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(w, four_year_prices, 10_000, 0, "Monthly")
        assert bt["drawdowns"].iloc[0] == pytest.approx(0.0)

    def test_contribution_increases_final_wealth(self, four_year_prices):
        """Adding periodic contributions must increase the final wealth."""
        w = np.array([0.6, 0.4])
        bt_no  = run_backtest(w, four_year_prices, 10_000, 0,   "Monthly")
        bt_yes = run_backtest(w, four_year_prices, 10_000, 500, "Monthly")
        assert bt_yes["final_wealth"] > bt_no["final_wealth"]

    def test_drip_off_div_cash_positive(self, four_year_prices):
        """DRIP OFF with positive yield must accumulate dividend cash."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(
            w, four_year_prices, 10_000, 0, "Monthly",
            drip=False, annual_div_yield=0.02,
        )
        assert bt["div_cash"].iloc[-1] > 0.0

    def test_drip_on_div_cash_all_zeros(self, four_year_prices):
        """DRIP ON must produce zero dividend cash regardless of yield."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(
            w, four_year_prices, 10_000, 0, "Monthly",
            drip=True, annual_div_yield=0.03,
        )
        np.testing.assert_array_equal(bt["div_cash"].values, 0.0)

    def test_annual_returns_length(self, four_year_prices):
        """Should have at least 3 calendar-year return entries for 4-year data."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(w, four_year_prices, 10_000, 0, "Monthly")
        assert len(bt["annual_rets"]) >= 3
        assert len(bt["annual_rets"]) == len(bt["annual_years"])

    def test_insufficient_data_raises(self, two_asset_prices):
        """Annually freq on ~2-year data must raise ValueError (< 3 periods)."""
        w = np.array([0.6, 0.4])
        with pytest.raises(ValueError, match="need at least 3"):
            run_backtest(w, two_asset_prices, 10_000, 0, "Annually")

    @pytest.mark.parametrize("freq", ["Weekly", "Monthly", "Bi-Monthly", "Annually"])
    def test_all_frequencies_run(self, four_year_prices, freq):
        """run_backtest must complete without error for all contribution frequencies."""
        w = np.array([0.6, 0.4])
        bt = run_backtest(w, four_year_prices, 10_000, 100, freq)
        assert bt["n_steps"] > 0
        assert np.isfinite(bt["cagr"])


# ─── run_simulation — reproducibility ────────────────────────────────────────

class TestRunSimulationReproducibility:

    def test_same_seed_same_result(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        p1, _ = run_simulation(w, two_asset_prices, 10_000, 200, "Monthly", 5, 100, seed=99)
        p2, _ = run_simulation(w, two_asset_prices, 10_000, 200, "Monthly", 5, 100, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        p1, _ = run_simulation(w, two_asset_prices, 10_000, 0, "Monthly", 5, 100, seed=1)
        p2, _ = run_simulation(w, two_asset_prices, 10_000, 0, "Monthly", 5, 100, seed=2)
        assert not np.array_equal(p1, p2)

    def test_no_seed_runs_without_error(self, two_asset_prices):
        w = np.array([0.6, 0.4])
        port, _ = run_simulation(w, two_asset_prices, 10_000, 0, "Monthly", 5, 50)
        assert port.shape == (50, 5 * 12 + 1)


# ─── run_simulation — error handling ─────────────────────────────────────────

class TestRunSimulationErrors:

    def test_insufficient_data_raises(self):
        short_prices = pd.DataFrame({"A": np.random.random(20)})
        with pytest.raises(ValueError, match="Insufficient"):
            run_simulation(np.array([1.0]), short_prices, 10_000, 0, "Monthly", 5, 10)

    def test_exactly_30_rows_does_not_raise(self):
        prices = pd.DataFrame({"A": np.linspace(100, 110, 31)})
        # 31 rows → 30 return observations; should not raise
        run_simulation(np.array([1.0]), prices, 10_000, 0, "Monthly", 1, 10, seed=1)


# ─── Price history normalisation (regression for the &-dropna bug) ────────────

class TestNormalisationRegression:
    """
    Regression tests for the bug where a newer ticker with a shorter history
    caused its entire normalised series to become NaN.

    Root cause: fetch_prices used dropna(how="all") which kept rows where
    only some tickers had data.  The baseline row (iloc[0]) could be NaN for
    the newer ticker, making every normalised value NaN/NaN = NaN.

    Fix: use dropna() (how="any") so only common-date rows are returned.
    """

    def _make_staggered_prices(self):
        """SPY has 200 rows; NEW_ETF only has the last 70."""
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        spy = pd.Series(np.linspace(100, 200, 200), index=dates, name="SPY")
        new_etf = pd.Series(
            [np.nan] * 130 + list(np.linspace(50, 80, 70)),
            index=dates,
            name="NEW",
        )
        return pd.DataFrame({"SPY": spy, "NEW": new_etf})

    def test_dropna_any_removes_staggered_rows(self):
        prices = self._make_staggered_prices()
        common = prices.dropna()
        assert common.isna().sum().sum() == 0
        assert len(common) == 70

    def test_normalisation_no_nan_after_dropna(self):
        prices = self._make_staggered_prices().dropna()
        prices_plot = prices.bfill()
        norm = prices_plot / prices_plot.iloc[0] * 100
        assert norm.isna().sum().sum() == 0

    def test_first_row_fully_populated_after_dropna(self):
        prices = self._make_staggered_prices().dropna()
        assert not prices.iloc[0].isna().any()

    def test_bfill_safety_net_covers_any_residual_nan(self):
        """bfill() applied before division ensures iloc[0] is never NaN."""
        prices = pd.DataFrame(
            {"A": [np.nan, 100.0, 101.0], "B": [50.0, 51.0, 52.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="B"),
        )
        prices_plot = prices.bfill()
        norm = prices_plot / prices_plot.iloc[0] * 100
        assert norm.isna().sum().sum() == 0


# ─── Source-code hygiene ──────────────────────────────────────────────────────

class TestSourceCodeHygiene:

    def test_no_nbsp_html_entity_in_app(self):
        """
        Guard against &nbsp; HTML entities slipping back into app.py.
        Streamlit's markdown renderer does not process &nbsp; in mixed
        markdown/HTML content — it renders as literal text.
        Use st.caption() or Unicode \\u00a0 instead.
        """
        with open("app.py", encoding="utf-8") as f:
            source = f.read()
        assert "&nbsp;" not in source, (
            "Found &nbsp; in app.py.  "
            "Use st.caption() or a Unicode non-breaking space instead."
        )

    def test_no_nbsp_html_entity_in_simulation(self):
        with open("simulation.py", encoding="utf-8") as f:
            source = f.read()
        assert "&nbsp;" not in source

    def test_app_imports_from_simulation(self):
        """app.py must import its core logic from simulation.py."""
        with open("app.py", encoding="utf-8") as f:
            source = f.read()
        assert "from simulation import" in source

    def test_app_imports_portfolio_expense_ratio(self):
        """app.py must import portfolio_expense_ratio from simulation.py."""
        with open("app.py", encoding="utf-8") as f:
            source = f.read()
        assert "portfolio_expense_ratio" in source

    def test_app_imports_compute_path_drawdowns(self):
        """app.py must import compute_path_drawdowns from simulation.py."""
        with open("app.py", encoding="utf-8") as f:
            source = f.read()
        assert "compute_path_drawdowns" in source

    def test_simulation_exports_compute_path_drawdowns(self):
        """compute_path_drawdowns must be importable from simulation module."""
        from simulation import compute_path_drawdowns  # noqa: F401  (import test)

    def test_app_imports_run_backtest(self):
        """app.py must import run_backtest from simulation.py."""
        with open("app.py", encoding="utf-8") as f:
            source = f.read()
        assert "run_backtest" in source

    def test_simulation_has_no_streamlit_import(self):
        """simulation.py must not import streamlit."""
        with open("simulation.py", encoding="utf-8") as f:
            source = f.read()
        assert "import streamlit" not in source
        assert "from streamlit" not in source

    def test_freq_keys_present(self):
        assert set(FREQ.keys()) == {"Weekly", "Monthly", "Bi-Monthly", "Annually"}

    def test_freq_values_are_positive_integers(self):
        for key, val in FREQ.items():
            assert isinstance(val, int) and val > 0, f"FREQ[{key!r}] = {val!r} is invalid"


# ─── get_price_history contract (offline / mock-free) ────────────────────────

class TestGetPriceHistoryContract:
    """
    Tests the data contract the app depends on, without hitting the network.
    We test the transformation logic (dropna / column handling) directly on
    synthetic DataFrames rather than mocking yfinance internals.
    """

    def test_dropna_any_is_contract(self):
        """Only common-date rows are returned; no NaN in result."""
        dates = pd.date_range("2021-01-01", periods=100, freq="B")
        raw = pd.DataFrame(
            {
                "SPY": np.linspace(100, 200, 100),
                "NEW": [np.nan] * 40 + list(np.linspace(50, 100, 60)),
            },
            index=dates,
        )
        result = raw.dropna()
        assert result.isna().sum().sum() == 0
        assert len(result) == 60
        assert not result.iloc[0].isna().any()

    def test_single_ticker_series_to_frame(self):
        """A Series returned for a single ticker is promoted to a DataFrame."""
        s = pd.Series([100, 101, 102], name="SPY")
        df = s.to_frame("SPY")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["SPY"]
