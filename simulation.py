"""
simulation.py
Pure business logic for the Monte Carlo Portfolio Simulator.
No Streamlit dependency — safe to import directly from tests.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Contribution frequency → number of compounding periods per year
FREQ: dict = {
    "Weekly": 52,
    "Monthly": 12,
    "Bi-Monthly": 6,
    "Annually": 1,
}

# Contribution frequency → pandas resample rule (period-end prices)
_BACKTEST_RESAMPLE: dict = {
    "Weekly":     "W",    # last trading day of each Mon–Sun week
    "Monthly":    "ME",   # last trading day of each month
    "Bi-Monthly": "2ME",  # last trading day of every two months
    "Annually":   "YE",   # last trading day of each year
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_ticker_info(ticker: str) -> Optional[dict]:
    """
    Fetch ticker metadata and current price from yfinance.

    Returns a dict with keys:
        ticker, name, price, type, currency, div_yield

    Returns None if the ticker cannot be found or has no price data.
    """
    try:
        info = yf.Ticker(ticker.upper()).info
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("navPrice")
            or info.get("previousClose")
        )
        if not price:
            return None

        div_yield = float(
            info.get("trailingAnnualDividendYield")
            or info.get("dividendYield")
            or 0.0
        )

        # ETF/fund expense ratio.  yfinance surfaces this as netExpenseRatio
        # (preferred) or annualReportExpenseRatio.  Stocks return 0.0.
        expense_ratio = float(
            info.get("netExpenseRatio")
            or info.get("annualReportExpenseRatio")
            or info.get("expenseRatio")
            or 0.0
        )

        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName") or ticker.upper(),
            "price": float(price),
            "type": info.get("quoteType", "N/A"),
            "currency": info.get("currency", "USD"),
            "div_yield": div_yield,
            "expense_ratio": expense_ratio,
        }
    except Exception:
        return None


def get_price_history(tickers: tuple, period: str) -> Optional[pd.DataFrame]:
    """
    Download dividend-adjusted (total-return) closing prices.

    Returns a DataFrame with ticker symbols as column names containing only
    rows where ALL requested tickers have data (the common date range).
    This guarantees that ``prices.iloc[0]`` is fully populated, which
    prevents silent NaN propagation during chart normalisation.

    Returns None on download failure or if the result is empty.
    """
    try:
        raw = yf.download(
            list(tickers), period=period, auto_adjust=True, progress=False
        )
        if raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
        else:
            # Single-ticker download returns plain OHLCV columns
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})

        # Drop rows where ANY ticker is missing.
        # Using the default how="any" (not how="all") ensures iloc[0] is
        # always fully populated for every column so chart normalisation
        # divides by a valid non-NaN baseline.
        return close.dropna()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def portfolio_div_yield(holdings: list) -> float:
    """
    Compute the portfolio-weighted annual dividend yield.

    Parameters
    ----------
    holdings : list of dicts, each expected to have 'value' (float) and
               optionally 'div_yield' (float, 0–1 scale, e.g. 0.015 for 1.5 %).

    Returns
    -------
    float in [0, 1].  Returns 0.0 for empty portfolios or missing keys.
    """
    total = sum(h.get("value", 0.0) for h in holdings)
    if total <= 0.0:
        return 0.0
    return sum(
        h.get("div_yield", 0.0) * h.get("value", 0.0) / total
        for h in holdings
    )


def portfolio_expense_ratio(holdings: list) -> float:
    """
    Compute the portfolio-weighted annual expense ratio.

    Parameters
    ----------
    holdings : list of dicts, each expected to have 'value' (float) and
               optionally 'expense_ratio' (float, 0–1 scale, e.g. 0.0009 for 0.09 %).

    Returns
    -------
    float in [0, 1].  Returns 0.0 for empty portfolios or missing keys.
    Stocks and assets without ER data contribute 0.0 to the weighted average.
    """
    total = sum(h.get("value", 0.0) for h in holdings)
    if total <= 0.0:
        return 0.0
    return sum(
        h.get("expense_ratio", 0.0) * h.get("value", 0.0) / total
        for h in holdings
    )


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------

def run_simulation(
    weights: np.ndarray,
    prices: pd.DataFrame,
    initial: float,
    contribution: float,
    freq: str,
    years: int,
    n_sims: int,
    drip: bool = True,
    annual_div_yield: float = 0.0,
    er_delta: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correlated GBM Monte Carlo simulation with optional DRIP support.

    The price history supplied in *prices* is assumed to be on a
    **total-return** (dividend-adjusted) basis (``auto_adjust=True`` from
    yfinance).  This means:

    * **DRIP ON** (default) — dividends are implicitly reinvested; the
      full total-return distribution is used.
    * **DRIP OFF** — the weighted dividend yield is stripped from the
      mean return each period, and dividends are accumulated separately
      rather than being added back to the portfolio.

    Because yfinance ``auto_adjust=True`` prices already have the fund's
    historical expense ratio deducted daily from NAV, the simulation is
    implicitly correct without any extra ER adjustment.  Use ``er_delta``
    only when you want to model a *different* ER from the one embedded in
    the historical data (e.g. what-if analysis or projecting a higher-cost
    share class).

    Parameters
    ----------
    weights          : portfolio weights, shape (n_assets,). Should sum to 1.
    prices           : historical adjusted-close DataFrame, columns = tickers.
    initial          : starting portfolio value ($).
    contribution     : per-period cash contribution ($).
    freq             : contribution frequency key from ``FREQ`` dict.
    years            : simulation horizon in years.
    n_sims           : number of independent simulation paths.
    drip             : True → dividends reinvested (total return).
                       False → dividends paid as cash, not reinvested.
    annual_div_yield : portfolio-weighted annual dividend yield (used only
                       when drip=False).  E.g. 0.015 for 1.5 %.
    er_delta         : additional annual expense-ratio drag beyond the ER
                       already embedded in historical prices.  Positive
                       values reduce expected return; negative values
                       increase it.  E.g. 0.005 adds −0.5 % p.a. drag.
                       Default 0.0 = use historical prices as-is.
    seed             : integer RNG seed for reproducibility; None = random.

    Returns
    -------
    portfolio_values : ndarray, shape (n_sims, total_steps + 1).
        Portfolio value at every time step.
        DRIP ON  → total-return basis (dividends reinvested).
        DRIP OFF → price-return basis only (dividends excluded).
    dividend_cash    : ndarray, shape (n_sims, total_steps + 1).
        Cumulative dividends paid out (all zeros when drip=True).

    Raises
    ------
    ValueError
        When ``prices`` has fewer than 30 observations after computing
        returns — too little data for reliable statistics.
    """
    steps_per_yr = FREQ[freq]
    total_steps = years * steps_per_yr
    days_per_step = 252.0 / steps_per_yr

    rets = prices.pct_change().dropna()
    if len(rets) < 30:
        raise ValueError(
            f"Insufficient historical data ({len(rets)} observations). "
            "Need at least 30 trading days to estimate return statistics."
        )

    mu = rets.mean().values * days_per_step

    # Apply ER delta to the per-period drift.
    # Subtracting er_delta/steps_per_yr from every asset's mu is equivalent to
    # reducing the portfolio return by er_delta/steps_per_yr each period
    # (because portfolio_return = weights @ asset_returns and weights sum to 1).
    if er_delta != 0.0:
        mu = mu - er_delta / steps_per_yr

    sigma = rets.cov().values * days_per_step
    sigma += np.eye(len(weights)) * 1e-9  # small jitter for numerical stability

    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        sigma += np.eye(len(weights)) * 1e-5
        L = np.linalg.cholesky(sigma)

    n_assets = len(weights)
    rng = np.random.default_rng(seed)

    portfolio_values = np.empty((n_sims, total_steps + 1))
    portfolio_values[:, 0] = initial
    dividend_cash = np.zeros((n_sims, total_steps + 1))

    # Per-period dividend rate stripped when DRIP is OFF.
    # When drip=True we always use total-return mu regardless of yield.
    div_per_period = 0.0
    if not drip and annual_div_yield > 0.0:
        div_per_period = annual_div_yield / steps_per_yr

    for t in range(1, total_steps + 1):
        z = rng.standard_normal((n_assets, n_sims))
        asset_rets = L @ z + mu[:, None]          # (n_assets, n_sims)
        port_ret = weights @ asset_rets            # (n_sims,)

        if div_per_period > 0.0:
            # Strip dividend component from total return → price-only return
            price_ret = port_ret - div_per_period
            # Payout is based on prior period portfolio value
            div_payout = np.maximum(0.0, portfolio_values[:, t - 1] * div_per_period)
            dividend_cash[:, t] = dividend_cash[:, t - 1] + div_payout
            portfolio_values[:, t] = (
                portfolio_values[:, t - 1] * (1.0 + price_ret) + contribution
            )
        else:
            # DRIP ON (or zero yield): use full total-return distribution
            portfolio_values[:, t] = (
                portfolio_values[:, t - 1] * (1.0 + port_ret) + contribution
            )

    return portfolio_values, dividend_cash


# ---------------------------------------------------------------------------
# Drawdown helpers
# ---------------------------------------------------------------------------

def compute_path_drawdowns(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Compute the running drawdown (fraction from running peak) for every path
    and every time step.

    For path *i* at time *t*:

        running_peak[i, t] = max(portfolio_values[i, 0 .. t])
        drawdown[i, t]     = (running_peak[i, t] - portfolio_values[i, t])
                             / running_peak[i, t]

    A value of 0 means the portfolio is at (or above) its previous all-time
    high.  A value of 0.20 means it is 20 % below its peak.

    Parameters
    ----------
    portfolio_values : ndarray, shape (n_sims, n_steps + 1).
        Total portfolio or wealth values at every time step.

    Returns
    -------
    drawdowns : ndarray, shape (n_sims, n_steps + 1), values in [0, 1].
    """
    # Running peak: cumulative max along the time axis (axis=1).
    running_peak = np.maximum.accumulate(portfolio_values, axis=1)
    # Guard against a zero initial value (pathological edge case).
    drawdowns = np.where(
        running_peak > 0,
        (running_peak - portfolio_values) / running_peak,
        0.0,
    )
    return drawdowns


# ---------------------------------------------------------------------------
# Historical backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    weights: np.ndarray,
    prices: pd.DataFrame,
    initial: float,
    contribution: float,
    freq: str,
    drip: bool = True,
    annual_div_yield: float = 0.0,
) -> dict:
    """
    Historical backtest: replay actual price data through the weighted portfolio.

    Unlike the Monte Carlo engine, this function uses the *real* price sequence
    rather than sampling from its statistical distribution.  The result shows
    how the portfolio would have behaved over the chosen historical window with
    the user's contribution and DRIP settings applied.

    Prices are assumed to be total-return (``auto_adjust=True``) so dividends
    are already embedded; the ``annual_div_yield`` / ``drip`` parameters mirror
    the Monte Carlo DRIP logic for consistency.

    Parameters
    ----------
    weights          : portfolio weights, shape (n_assets,).  Should sum to 1.
    prices           : historical adjusted-close DataFrame, columns = tickers.
    initial          : starting portfolio value ($).
    contribution     : per-period cash contribution ($).
    freq             : contribution frequency key from ``FREQ`` dict.
    drip             : True → dividends reinvested; False → paid out as cash.
    annual_div_yield : portfolio-weighted annual dividend yield (used only
                       when drip=False).

    Returns
    -------
    dict with keys
        portfolio_values     pd.Series — portfolio value at each period-end date
        div_cash             pd.Series — cumulative dividends paid out (DRIP OFF)
        total_wealth         pd.Series — portfolio_values + div_cash
        drawdowns            pd.Series — running drawdown fraction in [0, 1]
        total_invested       pd.Series — cumulative cash invested
        n_steps              int — number of contribution periods backtested
        n_years              float — horizon length in years
        total_invested_final float
        final_wealth         float
        cagr                 float — annualised daily-compounded return
        total_return_pct     float — (final_wealth / total_invested - 1)
        max_drawdown         float — worst peak-to-trough fraction
        ann_vol              float — annualised daily volatility
        sharpe               float — annualised Sharpe (rf = 0)
        annual_rets          np.ndarray — calendar-year portfolio returns
        annual_years         np.ndarray — corresponding calendar years (int)
        best_yr              float — best calendar-year return
        worst_yr             float — worst calendar-year return

    Raises
    ------
    ValueError
        When fewer than 3 resampled periods exist in the price history (need
        at least 2 return observations for a meaningful backtest).
    """
    rule = _BACKTEST_RESAMPLE[freq]
    steps_per_yr = FREQ[freq]

    # Period-end prices at the contribution frequency, dropping incomplete bins.
    prices_at_freq = prices.resample(rule).last().dropna()

    if len(prices_at_freq) < 3:
        raise ValueError(
            f"Only {len(prices_at_freq)} {freq.lower()} period(s) found in the "
            "selected history — need at least 3.  Choose a longer historical period."
        )

    # N-1 period returns from N period-end prices.
    period_rets = prices_at_freq.pct_change().dropna()
    n_steps = len(period_rets)

    div_per_period = (
        annual_div_yield / steps_per_yr
        if (not drip and annual_div_yield > 0.0)
        else 0.0
    )

    # ── Forward simulation through actual returns ──────────────────────────────
    portfolio_values = np.zeros(n_steps + 1)
    portfolio_values[0] = initial
    div_cash = np.zeros(n_steps + 1)

    for t in range(n_steps):
        port_ret = float(np.dot(weights, period_rets.iloc[t].values))
        if div_per_period > 0.0:
            price_ret = port_ret - div_per_period
            div_payout = max(0.0, portfolio_values[t] * div_per_period)
            div_cash[t + 1] = div_cash[t] + div_payout
            portfolio_values[t + 1] = (
                portfolio_values[t] * (1.0 + price_ret) + contribution
            )
        else:
            portfolio_values[t + 1] = (
                portfolio_values[t] * (1.0 + port_ret) + contribution
            )

    # Index: prices_at_freq has exactly n_steps + 1 rows → aligns perfectly.
    idx = prices_at_freq.index
    total_wealth = portfolio_values + div_cash
    total_invested_arr = initial + contribution * np.arange(n_steps + 1)

    # ── Drawdown ───────────────────────────────────────────────────────────────
    running_peak = np.maximum.accumulate(total_wealth)
    drawdowns = np.where(
        running_peak > 0,
        (running_peak - total_wealth) / running_peak,
        0.0,
    )

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    n_years = n_steps / steps_per_yr
    total_invested_final = float(total_invested_arr[-1])
    final_wealth = float(total_wealth[-1])
    total_return_pct = (
        (final_wealth / total_invested_final - 1.0)
        if total_invested_final > 0.0
        else 0.0
    )
    max_drawdown = float(drawdowns.max())

    # CAGR and Sharpe from daily returns (finer-grained than period returns).
    daily_rets = prices.pct_change().dropna()
    port_daily = daily_rets.values @ weights
    n_trading_days = len(port_daily)
    cagr = (
        float((1.0 + port_daily).prod() ** (252.0 / n_trading_days) - 1.0)
        if n_trading_days > 0
        else 0.0
    )
    daily_std = float(np.std(port_daily))
    ann_vol = daily_std * np.sqrt(252.0)
    sharpe = (
        float(np.mean(port_daily) / daily_std * np.sqrt(252.0))
        if daily_std > 0.0
        else 0.0
    )

    # Calendar-year returns (year-end-to-year-end from daily prices).
    try:
        annual_prices = prices.resample("YE").last().pct_change().dropna()
        annual_rets = (annual_prices.values @ weights).astype(float)
        annual_years = annual_prices.index.year.to_numpy()
    except Exception:
        annual_rets = np.array([], dtype=float)
        annual_years = np.array([], dtype=int)

    best_yr = float(annual_rets.max()) if len(annual_rets) > 0 else float("nan")
    worst_yr = float(annual_rets.min()) if len(annual_rets) > 0 else float("nan")

    return {
        "portfolio_values":     pd.Series(portfolio_values,   index=idx),
        "div_cash":             pd.Series(div_cash,           index=idx),
        "total_wealth":         pd.Series(total_wealth,       index=idx),
        "drawdowns":            pd.Series(drawdowns,          index=idx),
        "total_invested":       pd.Series(total_invested_arr, index=idx),
        "n_steps":              n_steps,
        "n_years":              n_years,
        "total_invested_final": total_invested_final,
        "final_wealth":         final_wealth,
        "cagr":                 cagr,
        "total_return_pct":     total_return_pct,
        "max_drawdown":         max_drawdown,
        "ann_vol":              ann_vol,
        "sharpe":               sharpe,
        "annual_rets":          annual_rets,
        "annual_years":         annual_years,
        "best_yr":              best_yr,
        "worst_yr":             worst_yr,
    }
